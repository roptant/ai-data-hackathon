//! `dictation-server` command line.
//!
//! ```text
//! dictation-server serve --root DIR [--listen 127.0.0.1:8787] [--allow-public-bind]
//! dictation-server tenant-create --root DIR --name NAME
//! dictation-server keygen --root DIR
//! dictation-server train --root DIR --tenant ID --trainer PROG [--trainer-arg A]...
//!     --base-model PATH --base-model-id ID --asr-worker PATH --regression-set DIR
//!     [--min-train N] [--min-heldout N] [--compute-seconds N]
//! dictation-server sweep --root DIR
//! ```
//!
//! Serve behind a TLS-terminating reverse proxy; this process speaks plain
//! HTTP and refuses non-loopback binds unless `--allow-public-bind` is given.

use std::{
    net::SocketAddr,
    path::PathBuf,
    process::ExitCode,
    sync::{Arc, Mutex},
    time::Duration,
};

use dictation_server::{
    http::{AppState, AuthThrottle, router, unix_now},
    store::ServerStore,
    training::{PromotionPolicy, TrainerConfig, describe, load_or_create_signing_key, train_tenant},
};

struct Arguments {
    values: Vec<(String, String)>,
    flags: Vec<String>,
}

impl Arguments {
    fn parse(items: impl Iterator<Item = String>) -> Self {
        let mut values = Vec::new();
        let mut flags = Vec::new();
        let mut items = items.peekable();
        while let Some(item) = items.next() {
            if let Some(name) = item.strip_prefix("--") {
                match items.peek() {
                    Some(next) if !next.starts_with("--") => {
                        values.push((name.to_owned(), items.next().unwrap_or_default()));
                    }
                    _ => flags.push(name.to_owned()),
                }
            }
        }
        Self { values, flags }
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.values.iter().find(|(key, _)| key == name).map(|(_, value)| value.as_str())
    }

    fn all(&self, name: &str) -> Vec<String> {
        self.values.iter().filter(|(key, _)| key == name).map(|(_, value)| value.clone()).collect()
    }

    fn require(&self, name: &str) -> Result<&str, String> {
        self.get(name).ok_or_else(|| format!("missing --{name}"))
    }
}

async fn serve(arguments: &Arguments) -> Result<(), String> {
    let root = PathBuf::from(arguments.require("root")?);
    let listen: SocketAddr = arguments
        .get("listen")
        .unwrap_or("127.0.0.1:8787")
        .parse()
        .map_err(|_| "invalid --listen address")?;
    if !listen.ip().is_loopback() && !arguments.flags.iter().any(|flag| flag == "allow-public-bind") {
        return Err("refusing non-loopback bind without --allow-public-bind (terminate TLS in front)".to_owned());
    }
    let store = Arc::new(Mutex::new(ServerStore::open(&root).map_err(|error| error.to_string())?));
    let sweeper = Arc::clone(&store);
    tokio::spawn(async move {
        loop {
            let store = Arc::clone(&sweeper);
            let _ = tokio::task::spawn_blocking(move || {
                if let Ok(mut store) = store.lock() {
                    let _ = store.sweep(unix_now());
                }
            })
            .await;
            tokio::time::sleep(Duration::from_secs(600)).await;
        }
    });
    let app = router(AppState {
        store,
        throttle: Arc::new(AuthThrottle::default()),
    });
    let listener = tokio::net::TcpListener::bind(listen).await.map_err(|error| error.to_string())?;
    println!("listening on {listen}");
    axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>())
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .map_err(|error| error.to_string())
}

fn run(command: &str, arguments: &Arguments) -> Result<(), String> {
    let root = PathBuf::from(arguments.require("root")?);
    match command {
        "tenant-create" => {
            let store = ServerStore::open(&root).map_err(|error| error.to_string())?;
            let (tenant, token) = store
                .create_tenant(arguments.require("name")?, unix_now())
                .map_err(|error| error.to_string())?;
            println!("tenant_id {}\ntoken {token}\n(the token is shown once; store it in the desktop app)", tenant.tenant_id);
            Ok(())
        }
        "keygen" => {
            let key = load_or_create_signing_key(&root.join("delivery-signing.key")).map_err(|error| error.to_string())?;
            println!("delivery public key {}", hex::encode(key.verifying_key().to_bytes()));
            Ok(())
        }
        "sweep" => {
            let mut store = ServerStore::open(&root).map_err(|error| error.to_string())?;
            let removed = store.sweep(unix_now()).map_err(|error| error.to_string())?;
            println!("removed {removed}");
            Ok(())
        }
        "train" => {
            let mut store = ServerStore::open(&root).map_err(|error| error.to_string())?;
            let parse = |name: &str, default: usize| -> Result<usize, String> {
                arguments.get(name).map_or(Ok(default), |value| value.parse().map_err(|_| format!("invalid --{name}")))
            };
            let defaults = PromotionPolicy::default();
            let policy = PromotionPolicy {
                min_training_examples: parse("min-train", defaults.min_training_examples)?,
                min_heldout_examples: parse("min-heldout", defaults.min_heldout_examples)?,
                ..defaults
            };
            let config = TrainerConfig {
                program: PathBuf::from(arguments.require("trainer")?),
                arguments: arguments.all("trainer-arg"),
                base_model: PathBuf::from(arguments.require("base-model")?),
                base_model_id: arguments.require("base-model-id")?.to_owned(),
                regression_set: arguments.get("regression-set").map(PathBuf::from),
                asr_worker: PathBuf::from(arguments.require("asr-worker")?),
                signing_key: load_or_create_signing_key(&root.join("delivery-signing.key")).map_err(|error| error.to_string())?,
            };
            let compute = Duration::from_secs(parse("compute-seconds", 4 * 3600)? as u64);
            let outcome = train_tenant(&mut store, arguments.require("tenant")?, &config, policy, compute, unix_now)
                .map_err(|error| error.to_string())?;
            println!("{}", describe(&outcome));
            Ok(())
        }
        _ => Err("usage: dictation-server <serve|tenant-create|keygen|train|sweep> --root DIR ...".to_owned()),
    }
}

fn main() -> ExitCode {
    let mut items = std::env::args().skip(1);
    let command = items.next().unwrap_or_default();
    let arguments = Arguments::parse(items);
    let result = if command == "serve" {
        match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
            Ok(runtime) => runtime.block_on(serve(&arguments)),
            Err(error) => Err(error.to_string()),
        }
    } else {
        run(&command, &arguments)
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
