//! Example caption client for the Local Dictation API (plan §8, milestone 2).
//!
//! ```text
//! caption_client pair --port 8765 --name "Caption overlay" --scopes transcript:live,transcript:final,status:read
//! caption_client captions --port 8765 --token-file token.txt [--once]
//! caption_client start|stop|cancel --port 8765 --token-file token.txt [--session ID]
//! ```
//!
//! Partial events replace the text of their segment (by id and revision);
//! they are never appended. `transcript.final` supersedes all partials.

use std::{collections::BTreeMap, time::Duration};

use futures_util::StreamExt;
use tokio_tungstenite::tungstenite::{client::IntoClientRequest, http::HeaderValue};

struct Options(Vec<(String, String)>, Vec<String>);

impl Options {
    fn get(&self, name: &str) -> Option<&str> {
        self.0.iter().find(|(key, _)| key == name).map(|(_, value)| value.as_str())
    }
    fn flag(&self, name: &str) -> bool {
        self.1.iter().any(|flag| flag == name)
    }
}

fn parse(arguments: impl Iterator<Item = String>) -> Options {
    let (mut values, mut flags) = (Vec::new(), Vec::new());
    let mut arguments = arguments.peekable();
    while let Some(argument) = arguments.next() {
        if let Some(name) = argument.strip_prefix("--") {
            match arguments.peek() {
                Some(next) if !next.starts_with("--") => values.push((name.to_owned(), arguments.next().unwrap_or_default())),
                _ => flags.push(name.to_owned()),
            }
        }
    }
    Options(values, flags)
}

fn token(options: &Options) -> Result<String, String> {
    let path = options.get("token-file").ok_or("missing --token-file")?;
    std::fs::read_to_string(path).map(|text| text.trim().to_owned()).map_err(|error| error.to_string())
}

async fn pair(base: &str, options: &Options) -> Result<(), String> {
    let http = reqwest::Client::builder().no_proxy().build().map_err(|error| error.to_string())?;
    let scopes: Vec<&str> = options.get("scopes").unwrap_or("transcript:live,transcript:final,status:read").split(',').collect();
    let opened: serde_json::Value = http
        .post(format!("{base}/v1/pairings"))
        .json(&serde_json::json!({ "client_name": options.get("name").unwrap_or("Caption client"), "scopes": scopes }))
        .send()
        .await
        .map_err(|error| error.to_string())?
        .json()
        .await
        .map_err(|error| error.to_string())?;
    let id = opened["pairing_id"].as_str().ok_or("pairing refused (rate limited?)")?;
    let secret = opened["poll_secret"].as_str().ok_or("missing poll secret")?;
    eprintln!("Approve \"{}\" in Local Dictation → Integrations. Verification code: {}", options.get("name").unwrap_or("Caption client"), opened["verification_code"]);
    for _ in 0..300 {
        tokio::time::sleep(Duration::from_secs(1)).await;
        let polled: serde_json::Value = http
            .get(format!("{base}/v1/pairings/{id}"))
            .header("x-pairing-secret", secret)
            .send()
            .await
            .map_err(|error| error.to_string())?
            .json()
            .await
            .map_err(|error| error.to_string())?;
        match polled["status"].as_str() {
            Some("approved") => {
                let token = polled["token"].as_str().ok_or("missing token")?;
                if let Some(path) = options.get("token-file") {
                    std::fs::write(path, token).map_err(|error| error.to_string())?;
                    eprintln!("Paired. Token saved to {path}.");
                } else {
                    println!("{token}");
                }
                return Ok(());
            }
            Some("denied") => return Err("pairing denied".to_owned()),
            Some("pending") => {}
            _ => return Err("pairing expired".to_owned()),
        }
    }
    Err("timed out waiting for approval".to_owned())
}

async fn captions(port: &str, options: &Options) -> Result<(), String> {
    let mut request = format!("ws://127.0.0.1:{port}/v1/events").into_client_request().map_err(|error| error.to_string())?;
    let header = HeaderValue::from_str(&format!("Bearer {}", token(options)?)).map_err(|error| error.to_string())?;
    request.headers_mut().insert("authorization", header);
    let (mut socket, _) = tokio_tungstenite::connect_async(request).await.map_err(|error| error.to_string())?;
    // segment id -> (revision, start_ms, text)
    let mut segments: BTreeMap<String, (u64, u64, String)> = BTreeMap::new();
    while let Some(message) = socket.next().await {
        let message = message.map_err(|error| error.to_string())?;
        let Ok(text) = message.to_text() else { continue };
        let Ok(event) = serde_json::from_str::<serde_json::Value>(text) else { continue };
        match event["event"].as_str() {
            Some("session.started") => {
                segments.clear();
                println!("[session started {}]", event["session_id"].as_str().unwrap_or(""));
            }
            Some("transcript.partial") => {
                let id = event["segment_id"].as_str().unwrap_or_default().to_owned();
                let revision = event["revision"].as_u64().unwrap_or(0);
                let stale = segments.get(&id).is_some_and(|(current, _, _)| *current >= revision);
                if !stale {
                    segments.insert(id, (revision, event["start_ms"].as_u64().unwrap_or(0), event["text"].as_str().unwrap_or_default().to_owned()));
                }
                let mut ordered: Vec<_> = segments.values().collect();
                ordered.sort_by_key(|(_, start, _)| *start);
                let caption: Vec<&str> = ordered.iter().map(|(_, _, text)| text.as_str()).collect();
                println!("partial: {}", caption.join(" "));
            }
            Some("transcript.final") => {
                segments.clear();
                println!("final: {}", event["text"].as_str().unwrap_or_default());
                if options.flag("once") {
                    return Ok(());
                }
            }
            Some("session.stopped") => println!("[session stopped]"),
            Some("session.cancelled") => {
                segments.clear();
                println!("[session cancelled; provisional captions withdrawn]");
            }
            Some("error") => {
                println!("[error {}]", event["code"].as_str().unwrap_or(""));
                if event["code"] == "resync_required" || event["code"] == "token_revoked" {
                    return Err(format!("stream closed: {}", event["code"]));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

async fn control(base: &str, command: &str, options: &Options) -> Result<(), String> {
    let http = reqwest::Client::builder().no_proxy().build().map_err(|error| error.to_string())?;
    let url = match command {
        "start" => format!("{base}/v1/sessions"),
        _ => format!("{base}/v1/sessions/{}/{command}", options.get("session").ok_or("missing --session")?),
    };
    let response = http.post(url).bearer_auth(token(options)?).send().await.map_err(|error| error.to_string())?;
    println!("{} {}", response.status().as_u16(), response.text().await.unwrap_or_default());
    Ok(())
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> std::process::ExitCode {
    let mut arguments = std::env::args().skip(1);
    let command = arguments.next().unwrap_or_default();
    let options = parse(arguments);
    let port = options.get("port").unwrap_or("8765").to_owned();
    let base = format!("http://127.0.0.1:{port}");
    let result = match command.as_str() {
        "pair" => pair(&base, &options).await,
        "captions" => captions(&port, &options).await,
        "start" | "stop" | "cancel" => control(&base, &command, &options).await,
        _ => Err("usage: caption_client <pair|captions|start|stop|cancel> [--port N] [--token-file F]".to_owned()),
    };
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}
