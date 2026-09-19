//! Application data directory layout (plan §6).
//!
//! Everything lives in the OS application-data directory, never a shared
//! temporary directory or the source tree. Directories are created owner-only
//! where the platform supports it and marked for exclusion from backups where
//! a convention exists.

use std::{
    fs, io,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataLayout {
    pub root: PathBuf,
}

impl DataLayout {
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Pinned, checksum-verified model weights. Not personal data.
    #[must_use]
    pub fn models(&self) -> PathBuf {
        self.root.join("models")
    }

    /// Encrypted raw session working data (inside the store database).
    #[must_use]
    pub fn sessions(&self) -> PathBuf {
        self.root.join("sessions")
    }

    /// Encrypted upload queue state.
    #[must_use]
    pub fn queue(&self) -> PathBuf {
        self.root.join("queue")
    }

    /// Content-free operational metadata.
    #[must_use]
    pub fn metadata(&self) -> PathBuf {
        self.root.join("metadata")
    }

    /// Raw session working data. Only the capture and training pipeline
    /// opens this database.
    #[must_use]
    pub fn sessions_database(&self) -> PathBuf {
        self.sessions().join("sessions.sqlite3")
    }

    /// Consent, jobs, receipts, and eligible packages. The upload worker
    /// opens only this database, so it has no path to raw recordings.
    #[must_use]
    pub fn queue_database(&self) -> PathBuf {
        self.queue().join("queue.sqlite3")
    }

    /// Content-free operational metadata and settings.
    #[must_use]
    pub fn metadata_database(&self) -> PathBuf {
        self.metadata().join("metadata.sqlite3")
    }

    /// Creates the layout and applies owner-only permissions and backup
    /// exclusion markers.
    ///
    /// # Errors
    ///
    /// Fails when a directory cannot be created or secured.
    pub fn ensure(&self) -> io::Result<()> {
        for directory in [
            self.root.clone(),
            self.models(),
            self.sessions(),
            self.queue(),
            self.metadata(),
        ] {
            fs::create_dir_all(&directory)?;
            restrict(&directory)?;
        }
        for directory in [self.sessions(), self.queue(), self.metadata()] {
            exclude_from_backup(&directory)?;
        }
        Ok(())
    }
}

#[cfg(unix)]
fn restrict(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
}

#[cfg(not(unix))]
fn restrict(_path: &Path) -> io::Result<()> {
    // Per-user application-data directories on Windows inherit the user's
    // profile ACL, which already excludes other users.
    Ok(())
}

/// Backup exclusion is best-effort: `CACHEDIR.TAG` is honored by several Linux
/// backup tools; macOS Time Machine honors `tmutil addexclusion`.
fn exclude_from_backup(path: &Path) -> io::Result<()> {
    let tag = path.join("CACHEDIR.TAG");
    if !tag.exists() {
        fs::write(
            &tag,
            "Signature: 8a477f597d28d172789f06886806bc55\n# Local Dictation encrypted working data; excluded from backups.\n",
        )?;
    }
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("/usr/bin/tmutil")
            .arg("addexclusion")
            .arg(path)
            .status();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_is_created_owner_only_with_backup_markers() {
        let root = std::env::temp_dir().join(format!("ld-layout-{}", std::process::id()));
        let layout = DataLayout::new(&root);
        layout.ensure().unwrap();
        assert!(layout.sessions().join("CACHEDIR.TAG").exists());
        assert!(!layout.models().join("CACHEDIR.TAG").exists());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(layout.sessions()).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o700);
        }
        fs::remove_dir_all(root).unwrap();
    }
}
