// Recording overlay. It never requests focus; state is conveyed by text and
// symbol, not color alone, and announced through an ARIA live region.

import { invoke, listen, type StatusView } from "./tauri.js";

const labels: Record<string, [string, string]> = {
  starting: ["…", "Starting"],
  recording: ["●", "Recording — release to insert"],
  recording_locked: ["🔒", "Recording (locked) — press toggle to stop"],
  finalizing: ["⋯", "Finishing"],
  delivering: ["⋯", "Inserting"],
  error: ["!", "Stopped"],
  cancelled: ["×", "Cancelled"],
  idle: ["", "Idle"],
};

const notices: Record<string, string> = {
  microphone_failure: "microphone failed",
  no_microphone: "no microphone found",
  microphone_permission_or_config: "microphone permission needed",
  asr_model_missing: "speech model not installed",
  transcription_failed: "transcription failed",
  no_speech_detected: "no speech detected",
  duration_limit_soon: "stopping soon (time limit)",
  cancelled_screen_locked: "cancelled: screen locked",
  cancelled_system_sleep: "cancelled: sleep",
};

const root = document.querySelector<HTMLElement>("#indicator");
const symbol = document.querySelector<HTMLElement>("#symbol");
const text = document.querySelector<HTMLElement>("#state");
const time = document.querySelector<HTMLElement>("#elapsed");
const caption = document.querySelector<HTMLElement>("#caption");

function format(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

function render(status: StatusView): void {
  if (!root || !symbol || !text || !time) {
    return;
  }
  const [icon, label] = labels[status.recordingState] ?? ["", status.recordingState];
  const notice = status.notice && notices[status.notice] ? ` — ${notices[status.notice]}` : "";
  root.dataset["state"] = status.recordingState;
  symbol.textContent = icon;
  text.textContent = `${label}${status.recordingState === "error" ? notice : ""}`;
  const recording = status.recordingState === "recording" || status.recordingState === "recording_locked";
  const remaining = status.maxSessionSeconds * 1000 - status.elapsedMs;
  time.textContent = recording ? `${format(status.elapsedMs)}${remaining < 30_000 ? " (ending soon)" : ""}` : "";
  if (!recording && caption) {
    caption.textContent = "";
  }
}

for (const button of document.querySelectorAll<HTMLButtonElement>("button[data-action]")) {
  button.addEventListener("click", () => {
    void invoke("recording_action", { action: button.dataset["action"] });
  });
}

void listen<StatusView>("status", render);
void listen<string>("caption", (value) => {
  if (caption) {
    caption.textContent = value;
  }
});
void invoke<StatusView>("get_status").then(render);
