// Minimal typed access to the Tauri API injected by `withGlobalTauri`.
// Only command invocation and event listening are used; the WebView has no
// filesystem, shell, or network permission.

interface TauriGlobal {
  core: { invoke<T>(command: string, args?: Record<string, unknown>): Promise<T> };
  event: {
    listen<T>(event: string, handler: (event: { payload: T }) => void): Promise<() => void>;
  };
}

declare global {
  interface Window {
    __TAURI__?: TauriGlobal;
  }
}

function tauri(): TauriGlobal {
  const api = window.__TAURI__;
  if (!api) {
    throw new Error("The desktop bridge is unavailable.");
  }
  return api;
}

export function invoke<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  return tauri().core.invoke<T>(command, args);
}

export function listen<T>(event: string, handler: (payload: T) => void): Promise<() => void> {
  return tauri().event.listen<T>(event, (message) => handler(message.payload));
}

export interface CapabilityView {
  microphone: string;
  globalShortcut: string;
  focusTracking: string;
  nativeInsertion: string;
  clipboard: string;
  nonactivatingIndicator: string;
  sleepWatch: boolean;
  screenLockWatch: boolean;
  session: string;
}

export interface StatusView {
  recordingState: string;
  sessionId: string | null;
  elapsedMs: number;
  maxSessionSeconds: number;
  locked: boolean;
  notice: string | null;
  asrReady: boolean;
  asrLoading: boolean;
  asrError: string | null;
  microphoneError: string | null;
  privacyModelReady: boolean;
  storageAvailable: boolean;
  contributionEnabled: boolean;
  trainingStatus: string;
  apiListening: number | null;
  shortcutsActive: boolean;
  shortcutTriggers: [string, string][];
  capabilities: CapabilityView | null;
}

export interface ShortcutSettings {
  hold: string;
  toggle: string;
  lock: string;
  cancel: string;
  cancel_portal: string;
}

export type ContributionTarget = "disabled" | "non_production" | "production";

export interface SettingsView {
  shortcuts: ShortcutSettings;
  automaticInsertion: boolean;
  clipboardFallbackDisclosed: boolean;
  excludedApplications: string[];
  privateTerms: string[];
  vocabulary: string[];
  language: string;
  asrModel: string;
  apiEnabled: boolean;
  apiPort: number;
  contributionTarget: ContributionTarget;
  serverUrl: string;
  serverTokenSet: boolean;
  deliveryPublicKey: string;
  maxSessionSeconds: number;
  onboardingComplete: boolean;
}

export interface ResultPanel {
  sessionId: string;
  text: string;
  reason: string;
}

export interface ModelInfo {
  id: string;
  role: "asr" | "privacy";
  displayName: string;
  sizeBytes: number;
  quantization: string;
  license: string;
  provenance: string;
  notes: string;
  measuredPeakRssMib: number;
  state: "missing" | "installed" | "corrupt";
}

export interface Disclosure {
  version: string;
  sections: [string, string][];
  gatesMet: boolean;
}

export interface History {
  consents: {
    consentId: string;
    version: string;
    grantedAt: number;
    expiresAt: number;
    revokedAt: number | null;
    paused: boolean;
  }[];
  jobs: { state: string; reason: string; durationMs: number; createdAt: number; updatedAt: number }[];
  pendingDeletions: number;
}

export interface Pairing {
  pairingId: string;
  clientName: string;
  requestedScopes: string[];
  verificationCode: string;
}

export interface Client {
  clientId: string;
  displayName: string;
  scopes: string[];
  createdAt: number;
  lastUsedAt: number | null;
  revoked: boolean;
}
