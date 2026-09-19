// Main window. Every dynamic value is inserted with textContent: pairing
// names come from untrusted local processes and must never become markup.

import {
  invoke,
  listen,
  type Client,
  type ContributionTarget,
  type Disclosure,
  type History,
  type ModelInfo,
  type Pairing,
  type ResultPanel,
  type SettingsView,
  type StatusView,
} from "./tauri.js";

type Child = Node | string | null | undefined | false;

function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attributes: Record<string, string | boolean | ((event: Event) => void)> = {},
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const element = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) {
    if (typeof value === "function") {
      element.addEventListener(name.replace(/^on/, ""), value);
    } else if (typeof value === "boolean") {
      if (value) element.setAttribute(name, "");
    } else {
      element.setAttribute(name, value);
    }
  }
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    element.append(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return element;
}

const app = document.querySelector<HTMLElement>("#app");
const live = document.querySelector<HTMLElement>("#live");
let status: StatusView | null = null;
let settings: SettingsView | null = null;
let page = "dictate";
let modelMessage = "";
let customSource = "";
let customChecksum = "";

function announce(message: string): void {
  if (live) live.textContent = message;
}

async function run(action: () => Promise<unknown>, success?: string): Promise<void> {
  try {
    await action();
    if (success) announce(success);
  } catch (error) {
    announce(String(error));
    window.alert(String(error));
  }
  await refresh();
}

const stateText: Record<string, string> = {
  idle: "Idle",
  starting: "Starting",
  recording: "Recording (holding)",
  recording_locked: "Recording (locked)",
  finalizing: "Finishing transcription",
  delivering: "Inserting",
  cancelled: "Cancelled",
  error: "Stopped with an error",
};

const capabilityText: Record<string, string> = {
  available: "Available",
  needs_permission: "Needs permission or setup",
  unavailable: "Not available",
};

function time(seconds: number | null): string {
  return seconds ? new Date(seconds * 1000).toLocaleString() : "—";
}

function megabytes(bytes: number): string {
  return `${(bytes / 1_048_576).toFixed(0)} MB`;
}

function section(title: string, ...children: Child[]): HTMLElement {
  return h("section", { class: "card" }, h("h2", {}, title), ...children);
}

function toggle(label: string, checked: boolean, onChange: (value: boolean) => void, help?: string): HTMLElement {
  const input = h("input", { type: "checkbox" });
  input.checked = checked;
  input.addEventListener("change", () => onChange(input.checked));
  return h("label", { class: "toggle" }, input, h("span", {}, label), help ? h("small", {}, help) : null);
}

function listEditor(label: string, items: string[], save: (items: string[]) => void, help: string): HTMLElement {
  const area = h("textarea", { rows: "4", "aria-describedby": `${label}-help` });
  area.value = items.join("\n");
  return h(
    "div",
    { class: "field" },
    h("label", {}, label, area),
    h("small", { id: `${label}-help` }, help),
    h("button", { onclick: () => save(area.value.split("\n")) }, `Save ${label.toLowerCase()}`),
  );
}

async function updateSettings(update: Record<string, unknown>, message = "Settings saved."): Promise<void> {
  await run(async () => {
    settings = await invoke<SettingsView>("update_settings", { update });
  }, message);
}

// ---------------------------------------------------------------- pages

async function dictatePage(): Promise<HTMLElement[]> {
  const result = await invoke<ResultPanel | null>("get_result");
  const recording = status?.recordingState === "recording" || status?.recordingState === "recording_locked";
  const triggers = status?.shortcutTriggers ?? [];
  return [
    section(
      "Dictation",
      h("p", { class: "state", role: "status" }, stateText[status?.recordingState ?? "idle"] ?? status?.recordingState ?? ""),
      status?.notice ? h("p", { class: "notice" }, status.notice.replaceAll("_", " ")) : null,
      status?.microphoneError ? h("p", { class: "notice", role: "alert" }, status.microphoneError) : null,
      h(
        "div",
        { class: "buttons" },
        h("button", { class: "primary", onclick: () => run(() => invoke("recording_action", { action: recording ? "stop" : "toggle" })) }, recording ? "Stop and insert" : "Start dictation"),
        h("button", { onclick: () => run(() => invoke("recording_action", { action: "cancel" })) }, "Cancel"),
      ),
      h(
        "p",
        {},
        status?.asrReady ? "Speech recognition is loaded." : status?.asrLoading ? "Loading speech recognition…" : "Speech recognition is not loaded yet; install a model under Setup.",
      ),
      triggers.length
        ? h("ul", { class: "triggers" }, ...triggers.map(([action, trigger]) => h("li", {}, `${action === "hold" ? "Hold down to record; release to stop" : action === "toggle" ? "Press once to start; press again to stop" : action}: ${trigger || "(set in system settings)"}`)))
        : h("p", { class: "notice" }, "Global shortcuts are not active. Use the buttons, tray menu, or Setup."),
      status?.asrError ? h("p", { class: "notice", role: "alert" }, status.asrError) : null,
    ),
    section(
      "Result not inserted automatically",
      result
        ? h(
            "div",
            {},
            h("p", {}, `Reason: ${result.reason.replaceAll("_", " ")}. Nothing was typed into another application.`),
            h("blockquote", {}, result.text),
            h(
              "div",
              { class: "buttons" },
              h("button", { class: "primary", onclick: () => run(() => invoke("copy_result"), "Copied. Paste where you want it.") }, "Copy to clipboard"),
              h("button", { onclick: () => run(() => invoke("dismiss_result"), "Discarded.") }, "Discard"),
              h("button", { onclick: () => run(() => invoke("do_not_contribute", { sessionId: result.sessionId }), "This session will not be contributed.") }, "Do not contribute this session"),
            ),
            h("small", {}, "Clipboard history or sync tools may retain copied text; the app cannot erase it from them."),
          )
        : h("p", {}, "No pending result."),
    ),
  ];
}

async function setupPage(): Promise<HTMLElement[]> {
  const models = await invoke<ModelInfo[]>("list_models");
  const current = settings;
  if (!current) return [];
  const source = h("input", { value: customSource, placeholder: "Local .bin file path or direct HTTPS download URL", oninput: (event) => { customSource = (event.target as HTMLInputElement).value; } });
  const checksum = h("input", { value: customChecksum, placeholder: "Required for URL downloads; optional for local files", oninput: (event) => { customChecksum = (event.target as HTMLInputElement).value; } });
  const shortcutInputs = Object.entries(current.shortcuts).map(([name, value]) => {
    const input = h("input", { value: String(value), "aria-label": `${name} shortcut` });
    return [name, input] as const;
  });
  return [
    section(
      "Speech and privacy models (run locally)",
      h("p", {}, "Downloads are explicit and verified against pinned checksums. After installation, dictation works offline."),
      h(
        "table",
        {},
        h("thead", {}, h("tr", {}, h("th", {}, "Model"), h("th", {}, "Size"), h("th", {}, "License"), h("th", {}, "State"), h("th", {}, ""))),
        h(
          "tbody",
          {},
          ...models.map((model) =>
            h(
              "tr",
              {},
              h("td", {}, h("strong", {}, model.displayName), h("br"), h("small", {}, `${model.role === "asr" ? "Recognition" : "Privacy filter"} · ${model.notes}`)),
              h("td", {}, megabytes(model.sizeBytes)),
              h("td", {}, model.license),
              h("td", {}, model.state),
              h(
                "td",
                {},
                model.state === "installed"
                  ? model.role === "asr"
                    ? h("button", { disabled: current.asrModel === model.id, onclick: () => updateSettings({ asrModel: model.id }, "Recognition model selected.") }, current.asrModel === model.id ? "In use" : "Use")
                    : "Installed"
                  : model.id === "custom-whisper" ? "Import the file again below"
                  : h("button", { onclick: () => run(() => invoke("install_model", { id: model.id }), "Download started.") }, "Download"),
              ),
            ),
          ),
        ),
      ),
      h("p", { id: "download-progress", role: "status" }, modelMessage),
      h("button", { onclick: () => run(() => invoke("cancel_model_download")) }, "Cancel installation"),
      status?.asrError ? h("p", { class: "notice", role: "alert" }, status.asrError) : null,
    ),
    section(
      "Custom speech model",
      h("p", {}, "Choose whisper.cpp GGML .bin weights. Hugging Face safetensors models need conversion first; a repository URL is not a model file. Importing copies the file into app storage and selects it for recognition."),
      h("label", { class: "field" }, "Model file or direct download URL", source),
      h("button", { onclick: () => { void invoke<string | null>("choose_model_file").then((path) => { if (path) { customSource = path; source.value = path; } }).catch((error: unknown) => window.alert(String(error))); } }, "Browse…"),
      h("label", { class: "field" }, "SHA-256", checksum),
      h("button", { onclick: () => run(() => invoke("install_custom_model", { source: source.value, sha256: checksum.value }), "Custom model installation started.") }, "Import / download and use"),
    ),
    section(
      "Automatic insertion",
      toggle(
        "Insert into the focused text field",
        current.automaticInsertion,
        (value) => updateSettings({ automaticInsertion: value }),
        "On Linux this turns on desktop accessibility so text can be inserted natively and focus can be verified. Text is inserted only if the field that was focused when you started is still focused. Password fields and excluded apps never receive text; Enter is never pressed.",
      ),
      toggle(
        "Allow the clipboard as a fallback",
        current.clipboardFallbackDisclosed,
        (value) => updateSettings({ clipboardFallbackDisclosed: value }),
        "Clipboard history and sync can keep the full dictation. The app asks the system to exclude it from history where possible and restores your previous clipboard only if it still holds the dictation.",
      ),
      listEditor("Excluded applications", current.excludedApplications, (items) => updateSettings({ excludedApplications: items }), "One application name per line; these never receive text automatically."),
    ),
    section(
      "Shortcuts",
      h("p", {}, "Hold to dictate, press toggle to start or stop, press lock while holding to keep recording. Escape cancels while recording (on Wayland the cancel chord is used instead)."),
      h("button", { onclick: () => updateSettings({ shortcuts: { ...current.shortcuts, hold: current.shortcuts.toggle, toggle: current.shortcuts.hold } }, "Hold and toggle shortcuts swapped.") }, `Use ${current.shortcuts.hold} as start/stop toggle`),
      ...shortcutInputs.map(([name, input]) => h("label", { class: "field" }, name.replace("_", " "), input)),
      h(
        "button",
        {
          onclick: () =>
            updateSettings({ shortcuts: Object.fromEntries(shortcutInputs.map(([name, input]) => [name, input.value])) }, "Shortcuts updated."),
        },
        "Save shortcuts",
      ),
      h("button", { onclick: () => run(() => invoke("retry_shortcuts"), "Shortcuts registered.") }, "Retry registration"),
    ),
    section(
      "Recognition",
      listEditor("Vocabulary", current.vocabulary, (items) => updateSettings({ vocabulary: items }), "Names and terms to recognize better. Stays on this device."),
      h(
        "label",
        { class: "field" },
        "Language (two-letter code, empty to detect)",
        (() => {
          const input = h("input", { value: current.language, maxlength: "2" });
          input.addEventListener("change", () => void updateSettings({ language: input.value }));
          return input;
        })(),
      ),
    ),
  ];
}

async function privacyPage(): Promise<HTMLElement[]> {
  const disclosure = await invoke<Disclosure>("get_disclosure");
  const history = status?.storageAvailable ? await invoke<History>("contribution_history").catch(() => null) : null;
  const current = settings;
  if (!current) return [];
  const active = history?.consents.find((consent) => consent.revokedAt === null && consent.expiresAt * 1000 > Date.now());
  const target = h("select", { "aria-label": "Contribution server" },
    h("option", { value: "disabled" }, "Off"),
    h("option", { value: "non_production" }, "Test server (non-production)"),
    h("option", { value: "production" }, "Production (blocked until evaluation gates are met)"),
  );
  target.value = current.contributionTarget;
  const url = h("input", { value: current.serverUrl, placeholder: "https://…", "aria-label": "Server address" });
  const token = h("input", { type: "password", placeholder: current.serverTokenSet ? "(saved)" : "Access token", "aria-label": "Access token" });
  const key = h("input", { value: current.deliveryPublicKey, placeholder: "Model signing key (hex)", "aria-label": "Model signing key" });
  const accepted = h("input", { type: "checkbox" });
  return [
    section(
      "Private terms",
      listEditor("Private terms", current.privateTerms, (items) => updateSettings({ privateTerms: items }), "Words that always make a sentence ineligible for training. They never leave this device."),
    ),
    section(
      "Contribute to your personal speech model (optional)",
      h("p", {}, "Off by default. Dictation keeps working the same whether or not you contribute."),
      !disclosure.gatesMet
        ? h("p", { class: "notice" }, "The privacy evaluation required for production uploads has not been completed, so uploads can only go to a test server.")
        : null,
      h("dl", { class: "disclosure" }, ...disclosure.sections.flatMap(([title, body]) => [h("dt", {}, title), h("dd", {}, body)])),
      active
        ? h(
            "div",
            {},
            h("p", {}, `Consent ${active.version} given ${time(active.grantedAt)}${active.paused ? " — paused" : ""}; renews ${time(active.expiresAt)}.`),
            h(
              "div",
              { class: "buttons" },
              h("button", { onclick: () => run(() => invoke("set_contribution_paused", { paused: !active.paused })) }, active.paused ? "Resume" : "Pause"),
              h("button", { onclick: () => window.confirm("Withdraw consent? Queued uploads are cancelled and deletion of received data starts.") && run(() => invoke("withdraw_consent"), "Consent withdrawn.") }, "Withdraw"),
            ),
          )
        : h(
            "div",
            {},
            h("label", { class: "toggle" }, accepted, h("span", {}, `I have read the text above (version ${disclosure.version}) and want to contribute.`)),
            h("button", { class: "primary", onclick: () => (accepted.checked ? run(() => invoke("grant_consent", { acceptedVersion: disclosure.version }), "Contribution enabled for future sessions.") : announce("Please confirm you have read the text.")) }, "Opt in"),
          ),
      h("h3", {}, "Server"),
      h("label", { class: "field" }, "Destination", target),
      h("label", { class: "field" }, "Address", url),
      h("label", { class: "field" }, "Access token", token),
      h("label", { class: "field" }, "Model signing key", key),
      h("button", {
        onclick: () => updateSettings({
          contributionTarget: target.value as ContributionTarget,
          serverUrl: url.value,
          serverToken: token.value,
          deliveryPublicKey: key.value,
        }),
      }, "Save server settings"),
      h(
        "p",
        {},
        h("button", { class: "danger", onclick: () => window.confirm("Delete your training data and personalized model? This cannot be undone.") && run(() => invoke("delete_training_data"), "Deletion requested.") }, "Delete my training data and personalized model"),
      ),
      h("small", {}, "Deleting examples removes the personalized model entirely; copies already exported elsewhere cannot be recalled."),
    ),
    section(
      "Contribution history (no content)",
      history
        ? h(
            "div",
            {},
            h("p", {}, `Status: ${status?.trainingStatus.replaceAll("_", " ") ?? "—"}. Pending deletion requests: ${history.pendingDeletions}.`),
            h(
              "table",
              {},
              h("thead", {}, h("tr", {}, h("th", {}, "Created"), h("th", {}, "State"), h("th", {}, "Reason"), h("th", {}, "Retained audio"))),
              h("tbody", {}, ...history.jobs.map((job) => h("tr", {}, h("td", {}, time(job.createdAt)), h("td", {}, job.state), h("td", {}, job.reason || "—"), h("td", {}, `${(job.durationMs / 1000).toFixed(1)} s`)))),
            ),
          )
        : h("p", {}, "Secure storage is unavailable, so contribution is disabled on this device."),
    ),
  ];
}

async function integrationsPage(): Promise<HTMLElement[]> {
  const current = settings;
  if (!current) return [];
  const pairings = await invoke<Pairing[]>("list_pairings");
  const clients = status?.storageAvailable ? await invoke<Client[]>("list_clients").catch(() => []) : [];
  return [
    section(
      "Local caption API",
      h("p", {}, "Lets approved apps on this computer show live captions or control recording. Off by default; listens on 127.0.0.1 only."),
      toggle("Enable the local API", current.apiEnabled, (value) => updateSettings({ apiEnabled: value })),
      h("p", {}, status?.apiListening ? `Listening on 127.0.0.1:${status.apiListening}.` : "Not listening."),
      h("small", {}, "Approved clients receive unredacted text. This app cannot control what they log or forward."),
    ),
    section(
      "Pairing requests",
      h("button", { onclick: () => { void refresh(); } }, "Refresh pairing requests"),
      pairings.length
        ? h(
            "div",
            {},
            ...pairings.map((pairing) => {
              const boxes = pairing.requestedScopes.map((scope) => {
                const box = h("input", { type: "checkbox" });
                box.checked = scope !== "session:control";
                return [scope, box] as const;
              });
              return h(
                "div",
                { class: "pairing" },
                h("p", {}, h("strong", {}, pairing.clientName), ` — code ${pairing.verificationCode} (compare with the requesting app)`),
                ...boxes.map(([scope, box]) => h("label", { class: "toggle" }, box, h("span", {}, scope))),
                h(
                  "div",
                  { class: "buttons" },
                  h("button", { class: "primary", onclick: () => run(() => invoke("approve_pairing", { pairingId: pairing.pairingId, scopes: boxes.filter(([, box]) => box.checked).map(([scope]) => scope) }), "Client approved.") }, "Approve"),
                  h("button", { onclick: () => run(() => invoke("deny_pairing", { pairingId: pairing.pairingId }), "Request denied.") }, "Deny"),
                ),
              );
            }),
          )
        : h("p", {}, "No pending requests."),
    ),
    section(
      "Paired clients",
      clients.length
        ? h(
            "table",
            {},
            h("thead", {}, h("tr", {}, h("th", {}, "Name"), h("th", {}, "Permissions"), h("th", {}, "Last used"), h("th", {}, ""))),
            h(
              "tbody",
              {},
              ...clients.map((client) =>
                h(
                  "tr",
                  {},
                  h("td", {}, client.displayName),
                  h("td", {}, client.scopes.join(", ")),
                  h("td", {}, time(client.lastUsedAt)),
                  h("td", {}, client.revoked ? "Revoked" : h("button", { onclick: () => run(() => invoke("revoke_client", { clientId: client.clientId }), "Access revoked.") }, "Revoke")),
                ),
              ),
            ),
          )
        : h("p", {}, "No paired clients."),
    ),
  ];
}

async function capabilitiesPage(): Promise<HTMLElement[]> {
  const capabilities = await invoke<StatusView["capabilities"]>("capability_matrix");
  if (!capabilities) return [section("Capabilities", h("p", {}, "Probing…"))];
  const rows: [string, string][] = [
    ["Microphone", capabilityText[capabilities.microphone] ?? capabilities.microphone],
    ["Global shortcuts", capabilityText[capabilities.globalShortcut] ?? capabilities.globalShortcut],
    ["Focus verification", capabilityText[capabilities.focusTracking] ?? capabilities.focusTracking],
    ["Automatic insertion", capabilityText[capabilities.nativeInsertion] ?? capabilities.nativeInsertion],
    ["Clipboard fallback", capabilityText[capabilities.clipboard] ?? capabilities.clipboard],
    ["Non-activating overlay", capabilityText[capabilities.nonactivatingIndicator] ?? capabilities.nonactivatingIndicator],
    ["Stops on sleep", capabilities.sleepWatch ? "Yes" : "Not detected"],
    ["Stops on screen lock", capabilities.screenLockWatch ? "Yes" : "Not detected"],
  ];
  return [
    section(
      `This computer (${capabilities.session})`,
      h("table", {}, h("tbody", {}, ...rows.map(([name, value]) => h("tr", {}, h("th", { scope: "row" }, name), h("td", {}, value))))),
      h("p", {}, "Support differs between operating systems and desktops. Where automatic insertion or focus verification is unavailable, results appear in this window for you to copy."),
    ),
  ];
}

const pages: Record<string, [string, () => Promise<HTMLElement[]>]> = {
  dictate: ["Dictate", dictatePage],
  setup: ["Setup", setupPage],
  privacy: ["Privacy & contribution", privacyPage],
  integrations: ["Integrations", integrationsPage],
  capabilities: ["Capabilities", capabilitiesPage],
};

async function refresh(): Promise<void> {
  if (!app) return;
  try {
    [status, settings] = await Promise.all([invoke<StatusView>("get_status"), invoke<SettingsView>("get_settings")]);
    const entry = pages[page] ?? pages["dictate"];
    if (!entry) return;
    const content = await entry[1]();
    const nav = h(
      "nav",
      { "aria-label": "Sections" },
      ...Object.entries(pages).map(([key, [label]]) =>
        h("button", { class: key === page ? "tab active" : "tab", "aria-current": key === page ? "page" : "false", onclick: () => { page = key; void refresh(); } }, label),
      ),
    );
    app.replaceChildren(h("header", {}, h("h1", {}, "Local Dictation")), nav, h("div", { class: "content" }, ...content));
  } catch (error) {
    app.replaceChildren(h("section", { class: "card" }, h("h1", {}, "Core unavailable"), h("p", {}, String(error))));
  }
}

let refreshTimer: number | undefined;
void listen<StatusView>("status", (next) => {
  const changed = next.recordingState !== status?.recordingState || next.notice !== status?.notice || next.asrReady !== status?.asrReady || next.asrError !== status?.asrError || next.asrLoading !== status?.asrLoading;
  status = next;
  if (changed) {
    announce(stateText[next.recordingState] ?? next.recordingState);
    window.clearTimeout(refreshTimer);
    refreshTimer = window.setTimeout(() => void refresh(), 100);
  }
});
void listen<string>("result-ready", () => void refresh());
void listen<[string, number, number]>("model-progress", ([id, received, total]) => {
  modelMessage = total ? `${id}: ${megabytes(received)} of ${megabytes(total)}` : `${id}: ${megabytes(received)} received`;
  const element = document.querySelector("#download-progress");
  if (element) element.textContent = modelMessage;
});
void listen<[string, string | null]>("model-installed", ([id, error]) => {
  modelMessage = error ? `${id} failed: ${error}` : `${id} installed and verified.`;
  announce(modelMessage);
  void refresh();
});
void refresh();
