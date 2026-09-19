//! AT-SPI focus tracking and native text insertion (Linux, X11 and Wayland).
//!
//! Toolkits expose their accessibility trees only while the session's
//! accessibility flag is on, so the flag is enabled only after the user turns
//! on automatic insertion. The service remembers the most recently focused
//! object; insertion goes through `EditableText.InsertText` at the caret,
//! never through synthesized key presses, and never into a password field.

use std::sync::{Arc, Mutex};

use atspi::{
    AccessibilityConnection, Role, State,
    events::object::StateChangedEvent,
    proxy::{accessible::ObjectRefExt, proxy_ext::ProxyExt},
};
use dictation_core::platform::FocusTarget;
use futures_util::StreamExt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FocusedObject {
    pub bus_name: String,
    pub path: String,
    pub application: String,
    pub role: Role,
    pub editable: bool,
}

impl FocusedObject {
    #[must_use]
    pub fn target(&self) -> FocusTarget {
        FocusTarget {
            application_id: self.application.clone(),
            window_id: self.bus_name.clone(),
            editable_id: Some(self.path.clone()),
            protected: self.role == Role::PasswordText,
        }
    }
}

#[derive(Debug)]
pub enum AccessibilityError {
    Unavailable(String),
    NoFocus,
    NotEditable,
    Protected,
    Refused,
}

impl std::fmt::Display for AccessibilityError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable(error) => write!(formatter, "accessibility unavailable: {error}"),
            Self::NoFocus => write!(formatter, "no focused accessible object"),
            Self::NotEditable => write!(formatter, "focused object is not editable text"),
            Self::Protected => write!(formatter, "focused object is a password field"),
            Self::Refused => write!(formatter, "application refused the insertion"),
        }
    }
}

impl std::error::Error for AccessibilityError {}

pub struct Accessibility {
    connection: AccessibilityConnection,
    focused: Arc<Mutex<Option<FocusedObject>>>,
}

impl Accessibility {
    /// Turns on the session accessibility flag and starts tracking focus.
    ///
    /// # Errors
    ///
    /// Fails when the AT-SPI bus is unavailable.
    pub async fn start() -> Result<Self, AccessibilityError> {
        atspi::connection::set_session_accessibility(true)
            .await
            .map_err(|error| AccessibilityError::Unavailable(error.to_string()))?;
        let connection = AccessibilityConnection::new()
            .await
            .map_err(|error| AccessibilityError::Unavailable(error.to_string()))?;
        connection
            .register_event::<StateChangedEvent>()
            .await
            .map_err(|error| AccessibilityError::Unavailable(error.to_string()))?;
        let focused = Arc::new(Mutex::new(None));
        let tracker = Arc::clone(&focused);
        let events = connection.event_stream();
        let bus = connection.connection().clone();
        tokio::spawn(async move {
            tokio::pin!(events);
            while let Some(event) = events.next().await {
                let Ok(event) = event else { continue };
                let Ok(changed) = StateChangedEvent::try_from(event) else { continue };
                if changed.state != State::Focused || !changed.enabled {
                    continue;
                }
                if let Some(object) = describe(&bus, &changed.item).await {
                    if let Ok(mut slot) = tracker.lock() {
                        *slot = Some(object);
                    }
                }
            }
        });
        Ok(Self { connection, focused })
    }

    /// The most recently focused object, if a toolkit reported one.
    #[must_use]
    pub fn focused(&self) -> Option<FocusedObject> {
        self.focused.lock().ok().and_then(|slot| slot.clone())
    }

    /// Inserts `text` at the caret of `expected`, which must still be the
    /// focused object. Returns an error instead of guessing.
    ///
    /// # Errors
    ///
    /// Fails when focus moved, the field is protected or not editable, or the
    /// application refuses.
    pub async fn insert(&self, expected: &FocusedObject, text: &str) -> Result<(), AccessibilityError> {
        let current = self.focused().ok_or(AccessibilityError::NoFocus)?;
        if &current != expected {
            return Err(AccessibilityError::NoFocus);
        }
        if current.role == Role::PasswordText {
            return Err(AccessibilityError::Protected);
        }
        let object = object_ref(&current.bus_name, &current.path).ok_or(AccessibilityError::NoFocus)?;
        let bus = self.connection.connection();
        let accessible = object
            .as_accessible_proxy(bus)
            .await
            .map_err(|_| AccessibilityError::NoFocus)?;
        let state = accessible.get_state().await.map_err(|_| AccessibilityError::NoFocus)?;
        if !state.contains(State::Focused) {
            return Err(AccessibilityError::NoFocus);
        }
        if !state.contains(State::Editable) {
            return Err(AccessibilityError::NotEditable);
        }
        let proxies = accessible.proxies().await.map_err(|_| AccessibilityError::NotEditable)?;
        let caret = proxies
            .text()
            .await
            .map_err(|_| AccessibilityError::NotEditable)?
            .caret_offset()
            .await
            .map_err(|_| AccessibilityError::NotEditable)?;
        let editable = proxies.editable_text().await.map_err(|_| AccessibilityError::NotEditable)?;
        let length = i32::try_from(text.chars().count()).map_err(|_| AccessibilityError::Refused)?;
        match editable.insert_text(caret.max(0), text, length).await {
            Ok(true) => Ok(()),
            _ => Err(AccessibilityError::Refused),
        }
    }
}

fn object_ref(bus_name: &str, path: &str) -> Option<atspi::ObjectRefOwned> {
    let name = zbus::names::UniqueName::try_from(bus_name.to_owned()).ok()?;
    let path = zbus::zvariant::ObjectPath::try_from(path.to_owned()).ok()?;
    Some(atspi::ObjectRef::new_owned(name, path))
}

async fn describe(bus: &zbus::Connection, item: &atspi::ObjectRefOwned) -> Option<FocusedObject> {
    let accessible = item.as_accessible_proxy(bus).await.ok()?;
    let role = accessible.get_role().await.ok()?;
    let state = accessible.get_state().await.ok()?;
    let application = accessible
        .get_application()
        .await
        .ok()
        .map(|app| app.name_as_str().unwrap_or_default().to_owned())
        .unwrap_or_default();
    let root = object_ref(&application, "/org/a11y/atspi/accessible/root")?;
    let application_name = match root.as_accessible_proxy(bus).await {
        Ok(root) => root.name().await.unwrap_or_default(),
        Err(_) => String::new(),
    };
    Some(FocusedObject {
        bus_name: item.name_as_str().unwrap_or_default().to_owned(),
        path: item.path_as_str().to_owned(),
        application: if application_name.is_empty() { application } else { application_name },
        role,
        editable: state.contains(State::Editable),
    })
}
