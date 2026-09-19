//! Remappable shortcut actions and their translation into recording events
//! (plan §5).
//!
//! Three actions plus cancel: hold-to-record, toggle, lock-current-recording,
//! and cancel. Key auto-repeat is ignored. Releasing the hold key is applied
//! only after a short grace window, because some platforms report the hold
//! release just before the lock chord (adding a modifier changes the chord).

use std::fmt;

use crate::recording::Event;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ShortcutAction {
    Hold,
    Toggle,
    Lock,
    Cancel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShortcutEvent {
    pub action: ShortcutAction,
    pub pressed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, PartialOrd, Ord, Hash)]
pub struct Modifiers {
    pub control: bool,
    pub alt: bool,
    pub shift: bool,
    pub super_key: bool,
}

/// A key chord such as `Ctrl+Alt+Space`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Chord {
    pub modifiers: Modifiers,
    /// Key name in the portable vocabulary: `Space`, `Period`, `Escape`,
    /// `A`–`Z`, `0`–`9`, `F1`–`F24`, `Return`, `BackSpace`, `Tab`, `Minus`.
    pub key: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChordError(pub String);

impl fmt::Display for ChordError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ChordError {}

const NAMED_KEYS: [&str; 10] = [
    "Space", "Period", "Escape", "Return", "BackSpace", "Tab", "Minus", "Comma", "Slash", "Backslash",
];

fn canonical_key(key: &str) -> Option<String> {
    if let Some(named) = NAMED_KEYS.iter().find(|name| name.eq_ignore_ascii_case(key)) {
        return Some((*named).to_owned());
    }
    let upper = key.to_ascii_uppercase();
    if upper.len() == 1 && upper.chars().all(|c| c.is_ascii_alphanumeric()) {
        return Some(upper);
    }
    if let Some(number) = upper.strip_prefix('F') {
        if number.parse::<u8>().is_ok_and(|value| (1..=24).contains(&value)) {
            return Some(upper);
        }
    }
    None
}

impl Chord {
    /// Parses `Ctrl+Alt+Space`-style text.
    ///
    /// # Errors
    ///
    /// Rejects unknown keys, repeated keys, and chords that are not usable as
    /// global shortcuts.
    pub fn parse(text: &str) -> Result<Self, ChordError> {
        let mut modifiers = Modifiers::default();
        let mut key = None;
        for part in text.split('+').map(str::trim).filter(|part| !part.is_empty()) {
            match part.to_ascii_lowercase().as_str() {
                "ctrl" | "control" => modifiers.control = true,
                "alt" => modifiers.alt = true,
                "shift" => modifiers.shift = true,
                "super" | "logo" | "meta" | "cmd" | "command" => modifiers.super_key = true,
                _ => {
                    if key.is_some() {
                        return Err(ChordError(format!("more than one key in {text:?}")));
                    }
                    key = Some(canonical_key(part).ok_or_else(|| ChordError(format!("unknown key {part:?}")))?);
                }
            }
        }
        let key = key.ok_or_else(|| ChordError(format!("no key in {text:?}")))?;
        let has_modifier = modifiers.control || modifiers.alt || modifiers.super_key;
        if !has_modifier && key != "Escape" && !key.starts_with('F') {
            return Err(ChordError(format!(
                "{text:?} needs Ctrl, Alt, or Super so ordinary typing is not captured"
            )));
        }
        Ok(Self { modifiers, key })
    }

    /// Human and `global-hotkey` format: `Ctrl+Alt+Space`.
    #[must_use]
    pub fn display(&self) -> String {
        let mut parts = Vec::new();
        if self.modifiers.control {
            parts.push("Ctrl");
        }
        if self.modifiers.alt {
            parts.push("Alt");
        }
        if self.modifiers.shift {
            parts.push("Shift");
        }
        if self.modifiers.super_key {
            parts.push("Super");
        }
        parts.push(&self.key);
        parts.join("+")
    }

    /// XDG shortcuts-specification trigger used by the portal.
    #[must_use]
    pub fn portal_trigger(&self) -> String {
        let mut parts = Vec::new();
        if self.modifiers.control {
            parts.push("CTRL".to_owned());
        }
        if self.modifiers.alt {
            parts.push("ALT".to_owned());
        }
        if self.modifiers.shift {
            parts.push("SHIFT".to_owned());
        }
        if self.modifiers.super_key {
            parts.push("LOGO".to_owned());
        }
        let key = match self.key.as_str() {
            "Space" => "space".to_owned(),
            "Period" => "period".to_owned(),
            "Minus" => "minus".to_owned(),
            "Comma" => "comma".to_owned(),
            "Slash" => "slash".to_owned(),
            "Backslash" => "backslash".to_owned(),
            other if other.len() == 1 => other.to_ascii_lowercase(),
            other => other.to_owned(),
        };
        parts.push(key);
        parts.join("+")
    }
}

/// User-remappable bindings. Provisional defaults avoid common desktop
/// shortcuts (e.g. Ctrl+Alt+L locks many desktops, Ctrl+Alt+T opens terminals).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShortcutBindings {
    pub hold: Chord,
    pub toggle: Chord,
    /// Pressed while holding: add Shift to the hold chord by default.
    pub lock: Chord,
    /// Registered only while recording on platforms that allow it.
    pub cancel: Chord,
    /// Static cancel chord where dynamic registration is impossible
    /// (Wayland portal), since grabbing Escape permanently would break apps.
    pub cancel_portal: Chord,
}

impl Default for ShortcutBindings {
    fn default() -> Self {
        let chord = |text: &str| Chord::parse(text).expect("default chord is valid");
        Self {
            hold: chord("Ctrl+Alt+Space"),
            toggle: chord("Ctrl+Alt+Period"),
            lock: chord("Ctrl+Alt+Shift+Space"),
            cancel: chord("Escape"),
            cancel_portal: chord("Ctrl+Alt+C"),
        }
    }
}

impl ShortcutBindings {
    /// # Errors
    ///
    /// Rejects duplicate chords across actions.
    pub fn validate(&self) -> Result<(), ChordError> {
        let chords = [&self.hold, &self.toggle, &self.lock, &self.cancel, &self.cancel_portal];
        for (index, chord) in chords.iter().enumerate() {
            if chords[index + 1..].contains(chord) {
                return Err(ChordError(format!("{} is assigned to two actions", chord.display())));
            }
        }
        Ok(())
    }
}

/// What the coordinator should do after one shortcut event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpreted {
    Apply(Event),
    /// Apply `HoldUp` at `deadline_ms` unless `Lock` arrives first.
    DeferHoldUp { deadline_ms: u64 },
    Ignore,
}

/// Grace window for a lock chord that arrives just after the hold release.
pub const HOLD_RELEASE_GRACE_MS: u64 = 200;

/// Stateful translation with auto-repeat suppression.
#[derive(Debug, Clone, Default)]
pub struct ShortcutInterpreter {
    held: [bool; 4],
    pending_hold_up: Option<u64>,
}

const fn slot(action: ShortcutAction) -> usize {
    match action {
        ShortcutAction::Hold => 0,
        ShortcutAction::Toggle => 1,
        ShortcutAction::Lock => 2,
        ShortcutAction::Cancel => 3,
    }
}

impl ShortcutInterpreter {
    pub fn handle(&mut self, event: ShortcutEvent, now_ms: u64) -> Interpreted {
        let index = slot(event.action);
        if event.pressed {
            if self.held[index] {
                return Interpreted::Ignore; // auto-repeat
            }
            self.held[index] = true;
            match event.action {
                ShortcutAction::Hold => {
                    self.pending_hold_up = None;
                    Interpreted::Apply(Event::HoldDown)
                }
                ShortcutAction::Toggle => Interpreted::Apply(Event::Toggle),
                ShortcutAction::Lock => {
                    // A deferred release is superseded by the lock.
                    self.pending_hold_up = None;
                    Interpreted::Apply(Event::Lock)
                }
                ShortcutAction::Cancel => Interpreted::Apply(Event::Cancel),
            }
        } else {
            self.held[index] = false;
            if event.action == ShortcutAction::Hold {
                let deadline_ms = now_ms + HOLD_RELEASE_GRACE_MS;
                self.pending_hold_up = Some(deadline_ms);
                Interpreted::DeferHoldUp { deadline_ms }
            } else {
                Interpreted::Ignore
            }
        }
    }

    /// Called periodically; yields the deferred `HoldUp` once due.
    pub fn poll(&mut self, now_ms: u64) -> Option<Event> {
        match self.pending_hold_up {
            Some(deadline) if now_ms >= deadline => {
                self.pending_hold_up = None;
                Some(Event::HoldUp)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn press(action: ShortcutAction) -> ShortcutEvent {
        ShortcutEvent { action, pressed: true }
    }

    fn release(action: ShortcutAction) -> ShortcutEvent {
        ShortcutEvent { action, pressed: false }
    }

    #[test]
    fn chords_parse_and_format_for_each_backend() {
        let chord = Chord::parse("ctrl + alt + space").unwrap();
        assert_eq!(chord.display(), "Ctrl+Alt+Space");
        assert_eq!(chord.portal_trigger(), "CTRL+ALT+space");
        assert!(Chord::parse("a").is_err(), "unmodified letters would capture typing");
        assert!(Chord::parse("Ctrl+Alt").is_err());
        assert!(Chord::parse("Ctrl+Nope").is_err());
        assert_eq!(Chord::parse("Escape").unwrap().key, "Escape");
    }

    #[test]
    fn defaults_are_valid_and_duplicates_are_rejected() {
        let bindings = ShortcutBindings::default();
        bindings.validate().unwrap();
        let mut duplicate = bindings;
        duplicate.toggle = duplicate.hold.clone();
        assert!(duplicate.validate().is_err());
    }

    #[test]
    fn auto_repeat_is_ignored() {
        let mut interpreter = ShortcutInterpreter::default();
        assert_eq!(interpreter.handle(press(ShortcutAction::Hold), 0), Interpreted::Apply(Event::HoldDown));
        assert_eq!(interpreter.handle(press(ShortcutAction::Hold), 30), Interpreted::Ignore);
    }

    #[test]
    fn a_late_lock_supersedes_the_hold_release() {
        let mut interpreter = ShortcutInterpreter::default();
        interpreter.handle(press(ShortcutAction::Hold), 0);
        assert_eq!(
            interpreter.handle(release(ShortcutAction::Hold), 1_000),
            Interpreted::DeferHoldUp { deadline_ms: 1_200 }
        );
        assert_eq!(interpreter.handle(press(ShortcutAction::Lock), 1_050), Interpreted::Apply(Event::Lock));
        assert_eq!(interpreter.poll(1_300), None);
    }

    #[test]
    fn a_plain_release_stops_after_the_grace_window() {
        let mut interpreter = ShortcutInterpreter::default();
        interpreter.handle(press(ShortcutAction::Hold), 0);
        interpreter.handle(release(ShortcutAction::Hold), 500);
        assert_eq!(interpreter.poll(600), None);
        assert_eq!(interpreter.poll(700), Some(Event::HoldUp));
        assert_eq!(interpreter.poll(800), None);
    }
}
