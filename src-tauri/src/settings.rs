//! User settings, persisted encrypted in the metadata store.
//!
//! Private terms and vocabulary stay local. Contribution server credentials
//! live here too, so they are protected by the OS-held master key.

use dictation_core::{
    contribution::UploadTarget,
    shortcuts::{Chord, ShortcutBindings},
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ShortcutSettings {
    pub hold: String,
    pub toggle: String,
    pub lock: String,
    pub cancel: String,
    pub cancel_portal: String,
}

impl Default for ShortcutSettings {
    fn default() -> Self {
        let bindings = ShortcutBindings::default();
        Self {
            hold: bindings.hold.display(),
            toggle: bindings.toggle.display(),
            lock: bindings.lock.display(),
            cancel: bindings.cancel.display(),
            cancel_portal: bindings.cancel_portal.display(),
        }
    }
}

impl ShortcutSettings {
    /// # Errors
    ///
    /// Returns a message naming the invalid or duplicate chord.
    pub fn bindings(&self) -> Result<ShortcutBindings, String> {
        let parse = |text: &str| Chord::parse(text).map_err(|error| error.to_string());
        let bindings = ShortcutBindings {
            hold: parse(&self.hold)?,
            toggle: parse(&self.toggle)?,
            lock: parse(&self.lock)?,
            cancel: parse(&self.cancel)?,
            cancel_portal: parse(&self.cancel_portal)?,
        };
        bindings.validate().map_err(|error| error.to_string())?;
        Ok(bindings)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ContributionTarget {
    #[default]
    Disabled,
    NonProduction,
    Production,
}

impl ContributionTarget {
    #[must_use]
    pub const fn upload_target(self) -> UploadTarget {
        match self {
            Self::Disabled => UploadTarget::Disabled,
            Self::NonProduction => UploadTarget::NonProduction,
            Self::Production => UploadTarget::Production,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct AppSettings {
    pub shortcuts: ShortcutSettings,
    /// Automatic insertion into the focused field. On Linux this enables the
    /// session accessibility bus.
    pub automatic_insertion: bool,
    /// The user has read that clipboard history/sync can retain dictation.
    pub clipboard_fallback_disclosed: bool,
    pub excluded_applications: Vec<String>,
    /// Words the rules always treat as sensitive. Never leave the device.
    pub private_terms: Vec<String>,
    /// Recognition hint for names and jargon. Never leaves the device.
    pub vocabulary: Vec<String>,
    /// ISO 639-1 code, or empty for detection.
    pub language: String,
    pub asr_model: String,
    pub api_enabled: bool,
    pub api_port: u16,
    pub contribution_target: ContributionTarget,
    pub server_url: String,
    pub server_token: String,
    /// Hex Ed25519 public key that signs personalized model manifests.
    pub delivery_public_key: String,
    pub max_session_seconds: u64,
    pub onboarding_complete: bool,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            shortcuts: ShortcutSettings::default(),
            automatic_insertion: false,
            clipboard_fallback_disclosed: false,
            excluded_applications: Vec::new(),
            private_terms: Vec::new(),
            vocabulary: Vec::new(),
            language: "en".to_owned(),
            asr_model: dictation_models::default_for(dictation_models::Role::Asr).id.to_owned(),
            api_enabled: false,
            api_port: 8765,
            contribution_target: ContributionTarget::Disabled,
            server_url: String::new(),
            server_token: String::new(),
            delivery_public_key: String::new(),
            max_session_seconds: 600,
            onboarding_complete: false,
        }
    }
}

/// View of settings safe to send to the `WebView`: the server token is replaced
/// by whether one is set.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingsView {
    pub shortcuts: ShortcutSettings,
    pub automatic_insertion: bool,
    pub clipboard_fallback_disclosed: bool,
    pub excluded_applications: Vec<String>,
    pub private_terms: Vec<String>,
    pub vocabulary: Vec<String>,
    pub language: String,
    pub asr_model: String,
    pub api_enabled: bool,
    pub api_port: u16,
    pub contribution_target: ContributionTarget,
    pub server_url: String,
    pub server_token_set: bool,
    pub delivery_public_key: String,
    pub max_session_seconds: u64,
    pub onboarding_complete: bool,
}

impl From<&AppSettings> for SettingsView {
    fn from(settings: &AppSettings) -> Self {
        Self {
            shortcuts: settings.shortcuts.clone(),
            automatic_insertion: settings.automatic_insertion,
            clipboard_fallback_disclosed: settings.clipboard_fallback_disclosed,
            excluded_applications: settings.excluded_applications.clone(),
            private_terms: settings.private_terms.clone(),
            vocabulary: settings.vocabulary.clone(),
            language: settings.language.clone(),
            asr_model: settings.asr_model.clone(),
            api_enabled: settings.api_enabled,
            api_port: settings.api_port,
            contribution_target: settings.contribution_target,
            server_url: settings.server_url.clone(),
            server_token_set: !settings.server_token.is_empty(),
            delivery_public_key: settings.delivery_public_key.clone(),
            max_session_seconds: settings.max_session_seconds,
            onboarding_complete: settings.onboarding_complete,
        }
    }
}

/// Partial update from the `WebView`. Omitted fields keep their value; the
/// token is replaced only when a new non-empty value is supplied.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct SettingsUpdate {
    pub shortcuts: Option<ShortcutSettings>,
    pub automatic_insertion: Option<bool>,
    pub clipboard_fallback_disclosed: Option<bool>,
    pub excluded_applications: Option<Vec<String>>,
    pub private_terms: Option<Vec<String>>,
    pub vocabulary: Option<Vec<String>>,
    pub language: Option<String>,
    pub asr_model: Option<String>,
    pub api_enabled: Option<bool>,
    pub api_port: Option<u16>,
    pub contribution_target: Option<ContributionTarget>,
    pub server_url: Option<String>,
    pub server_token: Option<String>,
    pub delivery_public_key: Option<String>,
    pub max_session_seconds: Option<u64>,
    pub onboarding_complete: Option<bool>,
}

fn clean_list(items: Vec<String>, maximum: usize) -> Vec<String> {
    let mut cleaned: Vec<String> = items
        .into_iter()
        .map(|item| item.trim().chars().filter(|c| !c.is_control()).take(80).collect::<String>())
        .filter(|item| !item.is_empty())
        .collect();
    cleaned.sort();
    cleaned.dedup();
    cleaned.truncate(maximum);
    cleaned
}

impl AppSettings {
    /// Applies an update after validating it.
    ///
    /// # Errors
    ///
    /// Returns a user-facing message for invalid values.
    pub fn apply(&mut self, update: SettingsUpdate) -> Result<(), String> {
        let mut next = self.clone();
        if let Some(shortcuts) = update.shortcuts {
            shortcuts.bindings()?;
            next.shortcuts = shortcuts;
        }
        if let Some(value) = update.automatic_insertion {
            next.automatic_insertion = value;
        }
        if let Some(value) = update.clipboard_fallback_disclosed {
            next.clipboard_fallback_disclosed = value;
        }
        if let Some(items) = update.excluded_applications {
            next.excluded_applications = clean_list(items, 200);
        }
        if let Some(items) = update.private_terms {
            next.private_terms = clean_list(items, 500);
        }
        if let Some(items) = update.vocabulary {
            next.vocabulary = clean_list(items, 200);
        }
        if let Some(language) = update.language {
            let language = language.trim().to_ascii_lowercase();
            if !(language.is_empty() || (language.len() == 2 && language.bytes().all(|b| b.is_ascii_lowercase()))) {
                return Err("Language must be a two-letter code or empty for detection.".to_owned());
            }
            next.language = language;
        }
        if let Some(model) = update.asr_model {
            let spec = dictation_models::spec(&model).ok_or("Unknown recognition model.")?;
            if spec.role != dictation_models::Role::Asr {
                return Err("That model is not a recognition model.".to_owned());
            }
            next.asr_model = model;
        }
        if let Some(value) = update.api_enabled {
            next.api_enabled = value;
        }
        if let Some(port) = update.api_port {
            if port < 1024 {
                return Err("Choose a port of 1024 or higher.".to_owned());
            }
            next.api_port = port;
        }
        if let Some(target) = update.contribution_target {
            next.contribution_target = target;
        }
        if let Some(url) = update.server_url {
            let url = url.trim().to_owned();
            if !(url.is_empty() || url.starts_with("https://") || url.starts_with("http://127.0.0.1:")) {
                return Err("The server must use HTTPS (loopback HTTP only for testing).".to_owned());
            }
            next.server_url = url;
        }
        if let Some(token) = update.server_token.filter(|token| !token.trim().is_empty()) {
            token.trim().clone_into(&mut next.server_token);
        }
        if let Some(key) = update.delivery_public_key {
            let key = key.trim().to_ascii_lowercase();
            if !(key.is_empty() || (key.len() == 64 && key.bytes().all(|b| b.is_ascii_hexdigit()))) {
                return Err("The delivery key must be 64 hexadecimal characters.".to_owned());
            }
            next.delivery_public_key = key;
        }
        if let Some(seconds) = update.max_session_seconds {
            next.max_session_seconds = seconds.clamp(60, 600);
        }
        if let Some(value) = update.onboarding_complete {
            next.onboarding_complete = value;
        }
        *self = next;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_updates_leave_settings_unchanged() {
        let mut settings = AppSettings::default();
        let before = settings.clone();
        let result = settings.apply(SettingsUpdate {
            vocabulary: Some(vec!["Kubernetes".to_owned()]),
            server_url: Some("http://example.com".to_owned()),
            ..SettingsUpdate::default()
        });
        assert!(result.is_err());
        assert_eq!(settings, before);
    }

    #[test]
    fn duplicate_shortcuts_are_rejected() {
        let mut settings = AppSettings::default();
        let mut shortcuts = ShortcutSettings::default();
        shortcuts.toggle = shortcuts.hold.clone();
        assert!(settings
            .apply(SettingsUpdate { shortcuts: Some(shortcuts), ..SettingsUpdate::default() })
            .is_err());
    }

    #[test]
    fn view_never_exposes_the_server_token() {
        let settings = AppSettings {
            server_token: "ldt_secret".to_owned(),
            ..AppSettings::default()
        };
        let view = serde_json::to_string(&SettingsView::from(&settings)).unwrap();
        assert!(!view.contains("ldt_secret"));
        assert!(view.contains("\"serverTokenSet\":true"));
    }
}
