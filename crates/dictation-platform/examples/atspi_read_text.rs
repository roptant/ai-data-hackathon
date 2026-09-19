//! Test helper: prints the text of the first editable text object in an
//! accessible application whose name contains the argument (e.g. `kwrite`).
//! Used by the end-to-end insertion check; Linux only.

#[cfg(target_os = "linux")]
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), String> {
    use atspi::{
        AccessibilityConnection, State,
        proxy::{accessible::ObjectRefExt, proxy_ext::ProxyExt},
    };

    let wanted = std::env::args().nth(1).ok_or("usage: atspi_read_text <application-name>")?.to_lowercase();
    let connection = AccessibilityConnection::new().await.map_err(|error| error.to_string())?;
    let bus = connection.connection();
    let root = connection.root_accessible_on_registry().await.map_err(|error| error.to_string())?;
    for application in root.get_children().await.map_err(|error| error.to_string())? {
        let Ok(proxy) = application.as_accessible_proxy(bus).await else { continue };
        let name = proxy.name().await.unwrap_or_default();
        if std::env::var_os("ATSPI_LIST").is_some() {
            eprintln!("application: {name}");
        }
        if !name.to_lowercase().contains(&wanted) {
            continue;
        }
        let mut stack = vec![application];
        let mut visited = 0;
        while let Some(object) = stack.pop() {
            visited += 1;
            if visited > 5_000 {
                break;
            }
            let Ok(proxy) = object.as_accessible_proxy(bus).await else { continue };
            let state = proxy.get_state().await.unwrap_or_default();
            if std::env::var_os("ATSPI_DUMP").is_some() {
                let role = proxy.get_role_name().await.unwrap_or_default();
                let interfaces = proxy.get_interfaces().await.map(|set| format!("{set:?}")).unwrap_or_default();
                eprintln!("{role} {state:?} {interfaces}");
            }
            if state.contains(State::Editable) {
                if let Ok(proxies) = proxy.proxies().await {
                    if let Ok(text) = proxies.text().await {
                        let count = text.character_count().await.unwrap_or(0);
                        let content = text.get_text(0, count).await.unwrap_or_default();
                        println!("{content}");
                        return Ok(());
                    }
                }
            }
            if let Ok(children) = proxy.get_children().await {
                stack.extend(children);
            }
        }
    }
    Err(format!("no editable text found in an application matching {wanted:?}"))
}

#[cfg(not(target_os = "linux"))]
fn main() {}
