//! Reports whether the OS credential store can hold the master key.
use dictation_storage::{MasterKeyProvider, keys::OsKeyring};

fn main() {
    let started = std::time::Instant::now();
    match OsKeyring.load_or_create() {
        Ok(Some(_)) => println!("master key available ({} ms)", started.elapsed().as_millis()),
        Ok(None) => println!("credential store unavailable ({} ms)", started.elapsed().as_millis()),
        Err(error) => println!("credential store error: {error}"),
    }
}
