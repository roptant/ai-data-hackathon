"""Application assembly.

One place that wires paths, settings, the encrypted store, consent, the queue,
the platform adapter and the two model workers together, so the CLI, the tests
and a future desktop shell all construct the same object graph.

Two assembly decisions carry the plan's intent:

* **No silent stand-ins.**  If a model role is unresolved or not installed,
  :meth:`Application.build_asr` and :meth:`Application.build_classifier` refuse.
  A scripted backend is used only when the caller passes ``allow_stub`` - the
  CLI does that for demos and says so on the console.
* **No storage without secure keys.**  Persistent payloads require the OS
  credential store.  Without it the application still dictates, with
  contribution storage disabled, rather than writing plaintext.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dictation.api.events import EventBus
from dictation.api.tokens import TokenManager
from dictation.asr.base import AsrWorker
from dictation.asr.mock import ScriptedAsr, script_from_text
from dictation.asr.whisper_cpp import WhisperCppWorker
from dictation.capture.coordinator import CaptureCoordinator
from dictation.config import Settings
from dictation.consent.consent import ConsentManager
from dictation.errors import ModelNotConfigured, ModelNotInstalled, SecureStorageUnavailable
from dictation.logging_ import events, set_install_salt
from dictation.models.fetch import verify_installed
from dictation.models.registry import ModelRegistry, ModelRole
from dictation.paths import DataPaths
from dictation.platform_.base import PlatformAdapter
from dictation.platform_.insertion import ResultPanel, TextDelivery
from dictation.platform_.probe import select_adapter
from dictation.privacy.classifier.base import PrivacyClassifier
from dictation.privacy.classifier.llama_cpp import LlamaCppPrivacyWorker
from dictation.privacy.classifier.mock import RuleEchoClassifier
from dictation.store.db import Database
from dictation.store.keys import EphemeralKeyStore, KeyStore, KeyringKeyStore
from dictation.store.retention import RetentionPolicy, run_cleanup
from dictation.store.session_store import SessionStore
from dictation.upload.queue import UploadQueue


@dataclass
class Application:
    """Assembled application services."""

    paths: DataPaths
    settings: Settings
    database: Database
    keys: KeyStore
    store: SessionStore
    consent: ConsentManager
    queue: UploadQueue
    tokens: TokenManager
    bus: EventBus
    registry: ModelRegistry
    adapter: PlatformAdapter
    retention: RetentionPolicy
    secure_storage: bool = True
    warnings: list[str] = field(default_factory=list)

    # -- construction --------------------------------------------------------

    @classmethod
    def open(
        cls,
        root: Path | None = None,
        *,
        allow_ephemeral_keys: bool = False,
        allow_unvalidated_upload: bool = False,
        adapter: PlatformAdapter | None = None,
        cleanup: bool = True,
    ) -> Application:
        paths = DataPaths.create(root)
        settings = Settings.load(paths.settings_file)
        database = Database.open(paths.database)
        warnings: list[str] = []

        secure = True
        # An explicit escape hatch for tests and CI, where touching the real
        # credential store would be a side effect on the developer's machine.
        forced_ephemeral = os.environ.get("DICTATION_EPHEMERAL_KEYS") == "1"
        try:
            if forced_ephemeral:
                raise SecureStorageUnavailable("DICTATION_EPHEMERAL_KEYS=1 is set")
            keys: KeyStore = KeyringKeyStore()
        except SecureStorageUnavailable as error:
            if not allow_ephemeral_keys:
                raise
            secure = False
            keys = EphemeralKeyStore()
            warnings.append(
                (
                    "ephemeral keys were requested; "
                    if forced_ephemeral
                    else f"no OS credential store ({error}); "
                )
                + "contribution storage is disabled and payloads will not survive a restart"
            )
        set_install_salt(keys.log_salt())

        store = SessionStore(paths, database, keys)
        consent = ConsentManager(
            database, allow_unvalidated_upload=allow_unvalidated_upload
        )
        retention = RetentionPolicy(settings.retention)
        queue = UploadQueue(
            database, store, consent, settings=settings.upload, retention=retention
        )
        tokens = TokenManager(
            database,
            auth_attempts_per_minute=settings.api.auth_attempts_per_minute,
            pairing_window_seconds=settings.api.pairing_window_seconds,
        )
        bus = EventBus(queue_limit=settings.api.client_queue_limit)
        registry = ModelRegistry.load(paths.models)

        if paths.inside_repository():
            warnings.append(
                "the payload root is inside a source checkout; move it outside before "
                "recording anything real"
            )

        application = cls(
            paths=paths,
            settings=settings,
            database=database,
            keys=keys,
            store=store,
            consent=consent,
            queue=queue,
            tokens=tokens,
            bus=bus,
            registry=registry,
            adapter=adapter or select_adapter(dry_run=True),
            retention=retention,
            secure_storage=secure,
            warnings=warnings,
        )
        if cleanup:
            run_cleanup(store, database)
        events.emit("app.opened", secure_storage=secure, warnings=len(warnings))
        return application

    def close(self) -> None:
        self.database.close()

    def __enter__(self) -> Application:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- model workers -------------------------------------------------------

    def build_asr(self, *, allow_stub: bool = False) -> AsrWorker:
        """Construct the ASR worker, or refuse with the reason."""
        try:
            spec = self.registry.chosen(ModelRole.ASR)
            path = self.registry.installed_path(spec)
        except (ModelNotConfigured, ModelNotInstalled):
            if not allow_stub:
                raise
            self.warnings.append(
                "no ASR model is configured; using a scripted stand-in that returns "
                "fixed text and does not recognise speech"
            )
            return ScriptedAsr(
                script_from_text(
                    "This is a scripted stand in transcript. "
                    "Choose and fetch an ASR model before relying on recognition."
                )
            )
        if not verify_installed(self.registry, spec):
            raise ModelNotInstalled(
                f"{spec.identifier} failed digest verification; refusing to load it"
            )
        return WhisperCppWorker(spec, path)

    def build_classifier(self, *, allow_stub: bool = False) -> PrivacyClassifier:
        """Construct the privacy worker, or refuse with the reason."""
        try:
            spec = self.registry.chosen(ModelRole.PRIVACY)
            path = self.registry.installed_path(spec)
        except (ModelNotConfigured, ModelNotInstalled):
            if not allow_stub:
                raise
            self.warnings.append(
                "no privacy model is configured; using the rule-echo stand-in, which "
                "finds only what the deterministic rules find and makes no contextual "
                "judgement. Automatic upload stays disabled."
            )
            return RuleEchoClassifier(private_terms=self.settings.private_terms)
        if not verify_installed(self.registry, spec):
            raise ModelNotInstalled(
                f"{spec.identifier} failed digest verification; refusing to load it"
            )
        return LlamaCppPrivacyWorker(spec, path)

    # -- coordinator ---------------------------------------------------------

    def build_coordinator(
        self,
        *,
        asr: AsrWorker | None = None,
        classifier: PrivacyClassifier | None = None,
        allow_stub: bool = False,
        contribution: bool = True,
    ) -> CaptureCoordinator:
        delivery = TextDelivery(self.adapter, self.settings.insertion, ResultPanel())
        return CaptureCoordinator(
            settings=self.settings,
            asr=asr or self.build_asr(allow_stub=allow_stub),
            delivery=delivery,
            adapter=self.adapter,
            classifier=classifier or self.build_classifier(allow_stub=allow_stub),
            bus=self.bus,
            store=self.store if self.secure_storage else None,
            queue=self.queue if (self.secure_storage and contribution) else None,
            consent=self.consent if (self.secure_storage and contribution) else None,
            retention=self.retention,
        )

    # -- settings ------------------------------------------------------------

    def save_settings(self, settings: Settings | None = None) -> None:
        if settings is not None:
            self.settings = settings
        self.settings.save(self.paths.settings_file)
