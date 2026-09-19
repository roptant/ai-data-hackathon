//! Active-window identity on X11 via EWMH `_NET_ACTIVE_WINDOW` and `WM_CLASS`.

use x11rb::{
    connection::Connection,
    protocol::xproto::{AtomEnum, ConnectionExt},
};

/// `(window id, WM_CLASS class)` of the focused top-level window.
#[must_use]
pub fn active_window() -> Option<(String, String)> {
    let (connection, screen) = x11rb::connect(None).ok()?;
    let root = connection.setup().roots.get(screen)?.root;
    let active_atom = connection.intern_atom(false, b"_NET_ACTIVE_WINDOW").ok()?.reply().ok()?.atom;
    let reply = connection
        .get_property(false, root, active_atom, AtomEnum::WINDOW, 0, 1)
        .ok()?
        .reply()
        .ok()?;
    let window = reply.value32()?.next()?;
    if window == 0 {
        return None;
    }
    let class = connection
        .get_property(false, window, AtomEnum::WM_CLASS, AtomEnum::STRING, 0, 256)
        .ok()?
        .reply()
        .ok()
        .map(|reply| {
            reply
                .value
                .split(|byte| *byte == 0)
                .filter(|part| !part.is_empty())
                .last()
                .map(|part| String::from_utf8_lossy(part).into_owned())
                .unwrap_or_default()
        })
        .unwrap_or_default();
    Some((format!("x11:{window:#x}"), class))
}
