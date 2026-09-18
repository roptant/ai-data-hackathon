"""Dataset and model lineage (plan section 10).

A directed graph from sample IDs through dataset versions, training jobs,
checkpoints, adapters, quantized builds and delivered artifacts.  Its purpose is
deletion: when a sample must go, the graph names everything derived from it, so
deletion covers queued uploads, object storage, derived datasets, checkpoints,
caches and affected model artifacts rather than only the row someone remembered.

Deleting examples does not remove their influence from trained weights.  The
first implementation therefore discards the whole customer adapter and falls
back to the base model; :meth:`Lineage.affected_artifacts` is what tells it
which adapters those are.  Reliable machine unlearning is not claimed.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path


class NodeKind(StrEnum):
    SAMPLE = "sample"
    DATASET = "dataset"
    JOB = "job"
    CHECKPOINT = "checkpoint"
    ADAPTER = "adapter"
    BUILD = "build"
    DELIVERY = "delivery"


@dataclass(frozen=True, slots=True)
class Node:
    node_id: str
    kind: NodeKind
    tenant_id: str
    created_at: float = field(default_factory=time.time)
    #: Non-content attributes only: sizes, versions, metric values.
    attributes: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["kind"] = str(self.kind)
        data["attributes"] = {key: value for key, value in self.attributes}
        return data


@dataclass
class Lineage:
    root: Path
    nodes: dict[str, Node] = field(default_factory=dict)
    #: child -> parents
    parents: dict[str, set[str]] = field(default_factory=dict)
    deleted: set[str] = field(default_factory=set)

    @property
    def path(self) -> Path:
        return self.root / "lineage.json"

    @classmethod
    def load(cls, root: Path) -> Lineage:
        lineage = cls(root=Path(root))
        try:
            raw = json.loads(lineage.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return lineage
        for entry in raw.get("nodes", []):
            node = Node(
                node_id=entry["node_id"],
                kind=NodeKind(entry["kind"]),
                tenant_id=entry["tenant_id"],
                created_at=entry.get("created_at", 0.0),
                attributes=tuple((k, str(v)) for k, v in (entry.get("attributes") or {}).items()),
            )
            lineage.nodes[node.node_id] = node
        for child, parents in (raw.get("parents") or {}).items():
            lineage.parents[child] = set(parents)
        lineage.deleted = set(raw.get("deleted") or [])
        return lineage

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "parents": {child: sorted(parents) for child, parents in self.parents.items()},
            "deleted": sorted(self.deleted),
        }
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)

    # -- writing -------------------------------------------------------------

    def add(
        self,
        node_id: str,
        kind: NodeKind,
        tenant_id: str,
        *,
        derived_from: tuple[str, ...] = (),
        attributes: dict[str, object] | None = None,
        now: float | None = None,
    ) -> Node:
        node = Node(
            node_id=node_id,
            kind=kind,
            tenant_id=tenant_id,
            created_at=time.time() if now is None else now,
            attributes=tuple((key, str(value)) for key, value in (attributes or {}).items()),
        )
        self.nodes[node_id] = node
        if derived_from:
            self.parents.setdefault(node_id, set()).update(derived_from)
        self.save()
        return node

    # -- queries -------------------------------------------------------------

    def children_of(self, node_id: str) -> tuple[str, ...]:
        return tuple(
            child for child, parents in self.parents.items() if node_id in parents
        )

    def descendants(self, node_id: str) -> tuple[str, ...]:
        """Everything derived from a node, transitively."""
        seen: set[str] = set()
        frontier = [node_id]
        while frontier:
            current = frontier.pop()
            for child in self.children_of(current):
                if child not in seen:
                    seen.add(child)
                    frontier.append(child)
        return tuple(sorted(seen))

    def ancestors(self, node_id: str) -> tuple[str, ...]:
        seen: set[str] = set()
        frontier = list(self.parents.get(node_id, ()))
        while frontier:
            current = frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            frontier.extend(self.parents.get(current, ()))
        return tuple(sorted(seen))

    def affected_artifacts(self, sample_ids: tuple[str, ...]) -> tuple[str, ...]:
        """Model artifacts that must be revoked when these samples are deleted."""
        affected: set[str] = set()
        for sample_id in sample_ids:
            for node_id in self.descendants(sample_id):
                node = self.nodes.get(node_id)
                if node and node.kind in {
                    NodeKind.ADAPTER,
                    NodeKind.CHECKPOINT,
                    NodeKind.BUILD,
                    NodeKind.DELIVERY,
                }:
                    affected.add(node_id)
        return tuple(sorted(affected))

    def samples_of(self, tenant_id: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                node.node_id
                for node in self.nodes.values()
                if node.tenant_id == tenant_id and node.kind is NodeKind.SAMPLE
            )
        )

    # -- deletion ------------------------------------------------------------

    def mark_deleted(self, node_ids: tuple[str, ...]) -> tuple[str, ...]:
        """Record deletion of nodes and everything derived from them."""
        removed: set[str] = set()
        for node_id in node_ids:
            removed.add(node_id)
            removed.update(self.descendants(node_id))
        self.deleted.update(removed)
        self.save()
        return tuple(sorted(removed))

    def is_deleted(self, node_id: str) -> bool:
        return node_id in self.deleted

    def restore_guard(self, node_id: str) -> None:
        """Raise if a restored backup tries to reintroduce a deleted sample.

        Backups are outside this process, so the guard is a check at ingest and
        at dataset-snapshot time rather than a promise about the backup system
        (plan section 10).
        """
        if self.is_deleted(node_id):
            raise ValueError(f"{node_id} was deleted and must not be reintroduced")
