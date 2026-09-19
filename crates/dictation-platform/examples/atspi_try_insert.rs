//! Diagnostic: inserts text into the first editable object of an application.

#[cfg(target_os = "linux")]
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), String> {
    use atspi::{AccessibilityConnection, State, proxy::{accessible::ObjectRefExt, proxy_ext::ProxyExt}};
    let wanted = std::env::args().nth(1).ok_or("usage: atspi_try_insert <app> <text>")?;
    let text = std::env::args().nth(2).unwrap_or_else(|| "hello".to_owned());
    let connection = AccessibilityConnection::new().await.map_err(|error| error.to_string())?;
    let bus = connection.connection();
    let root = connection.root_accessible_on_registry().await.map_err(|error| error.to_string())?;
    for application in root.get_children().await.map_err(|error| error.to_string())? {
        let Ok(proxy) = application.as_accessible_proxy(bus).await else { continue };
        if !proxy.name().await.unwrap_or_default().contains(&wanted) { continue; }
        let mut stack = vec![application];
        while let Some(object) = stack.pop() {
            let Ok(proxy) = object.as_accessible_proxy(bus).await else { continue };
            if proxy.get_state().await.unwrap_or_default().contains(State::Editable) {
                let proxies = proxy.proxies().await.map_err(|e| e.to_string())?;
                let caret = proxies.text().await.map_err(|e| e.to_string())?.caret_offset().await;
                println!("caret: {caret:?}");
                let editable = proxies.editable_text().await.map_err(|e| e.to_string())?;
                let length = i32::try_from(text.chars().count()).unwrap_or(0);
                println!("insert_text: {:?}", editable.insert_text(caret.unwrap_or(0).max(0), &text, length).await);
                println!("set_text_contents: {:?}", editable.set_text_contents(&text).await);
                return Ok(());
            }
            if let Ok(children) = proxy.get_children().await { stack.extend(children); }
        }
    }
    Err("not found".to_owned())
}

#[cfg(not(target_os = "linux"))]
fn main() {}
