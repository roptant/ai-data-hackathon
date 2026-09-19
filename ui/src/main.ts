import { invoke } from "@tauri-apps/api/core";
import "./styles.css";

interface ShellStatus {
  appVersion: string;
  migrationPhase: string;
  recordingState: string;
  components: {
    microphoneReady: boolean;
    asrReady: boolean;
    privacyModelReady: boolean;
  };
  contributionEnabled: boolean;
  notice: string;
}

const app = document.querySelector<HTMLElement>("#app");

if (!app) {
  throw new Error("missing application root");
}

function readiness(value: boolean): string {
  return value ? "Ready" : "Not wired";
}

function render(status: ShellStatus): void {
  app.innerHTML = `
    <section class="shell">
      <header>
        <p class="eyebrow">Rust + Tauri migration</p>
        <h1>Local Dictation</h1>
        <p class="notice">${status.notice}</p>
      </header>
      <div class="state-card">
        <span class="pulse" aria-hidden="true"></span>
        <div>
          <p class="label">Core state</p>
          <p class="state">${status.recordingState}</p>
        </div>
        <span class="phase">${status.migrationPhase}</span>
      </div>
      <dl class="readiness">
        <div><dt>Microphone capture</dt><dd>${readiness(status.components.microphoneReady)}</dd></div>
        <div><dt>Local ASR</dt><dd>${readiness(status.components.asrReady)}</dd></div>
        <div><dt>Privacy model</dt><dd>${readiness(status.components.privacyModelReady)}</dd></div>
        <div><dt>Contribution</dt><dd>${status.contributionEnabled ? "Enabled" : "Disabled"}</dd></div>
      </dl>
      <footer>Version ${status.appVersion} · No microphone or upload permission is exposed to this WebView.</footer>
    </section>
  `;
}

function renderFailure(): void {
  app.innerHTML = `
    <section class="shell error">
      <p class="eyebrow">Rust + Tauri migration</p>
      <h1>Core unavailable</h1>
      <p>The desktop command boundary did not respond. Recording and contribution remain disabled.</p>
    </section>
  `;
}

invoke<ShellStatus>("shell_status").then(render).catch(renderFailure);
