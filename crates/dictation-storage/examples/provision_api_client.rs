//! Development helper: registers a paired API client directly in the app's
//! encrypted metadata store (equivalent to approving a pairing in the UI).
//! Requires access to the user's own OS keyring.
//!
//! `provision_api_client <metadata.sqlite3> <name> <token> <scope>...`

use dictation_storage::{EncryptedStore, api_clients::Scope, keys::OsKeyring};

fn main() -> Result<(), String> {
    let mut arguments = std::env::args().skip(1);
    let database = arguments.next().ok_or("missing database path")?;
    let name = arguments.next().ok_or("missing name")?;
    let token = arguments.next().ok_or("missing token")?;
    let scopes: Option<Vec<Scope>> = arguments.map(|scope| Scope::parse(&scope)).collect();
    let scopes = scopes.ok_or("unknown scope")?;
    let store = EncryptedStore::open(std::path::Path::new(&database), &OsKeyring).map_err(|error| error.to_string())?;
    let now = i64::try_from(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_err(|error| error.to_string())?.as_secs()).unwrap_or(0);
    store
        .insert_api_client(&format!("client-dev-{now}"), &name, &token, &scopes, now)
        .map_err(|error| error.to_string())?;
    println!("provisioned {name}");
    Ok(())
}
