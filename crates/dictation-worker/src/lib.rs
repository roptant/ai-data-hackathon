//! Length-bounded private standard-I/O protocol for model workers.
//!
//! Audio is sent through an inherited pipe rather than a plaintext temporary
//! file. Each request gets a fresh process with a cleared environment and a
//! mandatory deadline; timeout kills the child. [`persistent::PersistentWorker`]
//! keeps a loaded model across requests with the same deadline semantics.

pub mod messages;
pub mod persistent;
pub mod sandbox;
pub mod serve;

use std::{
    ffi::OsString,
    fmt,
    io::{self, Read, Write},
    path::PathBuf,
    process::{Command, Stdio},
    sync::{Arc, Mutex, mpsc},
    thread,
    time::Duration,
};

const MAGIC: [u8; 4] = *b"LDIP";
const VERSION: u8 = 1;
const HEADER_LEN: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameKind {
    ControlJson = 1,
    PcmS16Le = 2,
}

impl TryFrom<u8> for FrameKind {
    type Error = ProtocolError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::ControlJson),
            2 => Ok(Self::PcmS16Le),
            _ => Err(ProtocolError::InvalidFrameKind(value)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub kind: FrameKind,
    pub payload: Vec<u8>,
}

#[derive(Debug)]
pub enum ProtocolError {
    Io(io::Error),
    InvalidMagic,
    UnsupportedVersion(u8),
    InvalidFrameKind(u8),
    FrameTooLarge { declared: usize, maximum: usize },
    WorkerSpawn(io::Error),
    MissingPipe,
    Timeout,
    WorkerFailed,
    PoisonedProcessLock,
}

impl fmt::Display for ProtocolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "worker IPC failed: {error}"),
            Self::InvalidMagic => write!(formatter, "worker frame has invalid magic"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported worker protocol version {version}")
            }
            Self::InvalidFrameKind(kind) => write!(formatter, "invalid worker frame kind {kind}"),
            Self::FrameTooLarge { declared, maximum } => {
                write!(formatter, "worker frame size {declared} exceeds {maximum}")
            }
            Self::WorkerSpawn(error) => write!(formatter, "worker could not start: {error}"),
            Self::MissingPipe => write!(formatter, "worker pipe was not created"),
            Self::Timeout => write!(formatter, "worker request timed out"),
            Self::WorkerFailed => write!(formatter, "worker exited unsuccessfully"),
            Self::PoisonedProcessLock => write!(formatter, "worker process lock was poisoned"),
        }
    }
}

impl std::error::Error for ProtocolError {}

impl From<io::Error> for ProtocolError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// Writes one bounded protocol frame.
///
/// # Errors
///
/// Returns an error when the payload exceeds the configured maximum, cannot
/// fit the wire length field, or the writer fails.
pub fn write_frame(
    writer: &mut impl Write,
    kind: FrameKind,
    payload: &[u8],
    maximum: usize,
) -> Result<(), ProtocolError> {
    if payload.len() > maximum {
        return Err(ProtocolError::FrameTooLarge {
            declared: payload.len(),
            maximum,
        });
    }
    let length = u32::try_from(payload.len()).map_err(|_| ProtocolError::FrameTooLarge {
        declared: payload.len(),
        maximum: u32::MAX as usize,
    })?;
    let mut header = [0_u8; HEADER_LEN];
    header[0..4].copy_from_slice(&MAGIC);
    header[4] = VERSION;
    header[5] = kind as u8;
    header[6..10].copy_from_slice(&length.to_be_bytes());
    writer.write_all(&header)?;
    writer.write_all(payload)?;
    Ok(())
}

/// Reads one frame without allocating beyond the configured maximum.
///
/// # Errors
///
/// Returns an error for truncated input, invalid protocol fields, oversized
/// payload declarations, or reader failures.
pub fn read_frame(reader: &mut impl Read, maximum: usize) -> Result<Frame, ProtocolError> {
    let mut header = [0_u8; HEADER_LEN];
    reader.read_exact(&mut header)?;
    if header[0..4] != MAGIC {
        return Err(ProtocolError::InvalidMagic);
    }
    if header[4] != VERSION {
        return Err(ProtocolError::UnsupportedVersion(header[4]));
    }
    let kind = FrameKind::try_from(header[5])?;
    let declared = usize::try_from(u32::from_be_bytes([
        header[6], header[7], header[8], header[9],
    ]))
    .unwrap_or(usize::MAX);
    if declared > maximum {
        return Err(ProtocolError::FrameTooLarge { declared, maximum });
    }
    let mut payload = vec![0_u8; declared];
    reader.read_exact(&mut payload)?;
    Ok(Frame { kind, payload })
}

#[derive(Debug, Clone)]
pub struct WorkerRunner {
    pub executable: PathBuf,
    pub arguments: Vec<OsString>,
    pub working_directory: PathBuf,
    pub maximum_control_bytes: usize,
    pub maximum_audio_bytes: usize,
}

impl WorkerRunner {
    /// Sends control JSON and optional PCM to a fresh private worker process.
    ///
    /// # Errors
    ///
    /// Returns an error for spawn/pipe failures, bounded-protocol violations,
    /// timeout, or unsuccessful worker exit.
    pub fn request(
        &self,
        control_json: &[u8],
        pcm_s16le: Option<&[u8]>,
        timeout: Duration,
    ) -> Result<Vec<u8>, ProtocolError> {
        if control_json.len() > self.maximum_control_bytes {
            return Err(ProtocolError::FrameTooLarge {
                declared: control_json.len(),
                maximum: self.maximum_control_bytes,
            });
        }
        if pcm_s16le.is_some_and(|pcm| pcm.len() > self.maximum_audio_bytes) {
            return Err(ProtocolError::FrameTooLarge {
                declared: pcm_s16le.map_or(0, <[u8]>::len),
                maximum: self.maximum_audio_bytes,
            });
        }
        let mut child = Command::new(&self.executable)
            .args(&self.arguments)
            .current_dir(&self.working_directory)
            .env_clear()
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(ProtocolError::WorkerSpawn)?;
        let mut stdin = child.stdin.take().ok_or(ProtocolError::MissingPipe)?;
        let mut stdout = child.stdout.take().ok_or(ProtocolError::MissingPipe)?;
        write_frame(
            &mut stdin,
            FrameKind::ControlJson,
            control_json,
            self.maximum_control_bytes,
        )?;
        if let Some(pcm) = pcm_s16le {
            write_frame(
                &mut stdin,
                FrameKind::PcmS16Le,
                pcm,
                self.maximum_audio_bytes,
            )?;
        }
        stdin.flush()?;
        drop(stdin);

        let child = Arc::new(Mutex::new(child));
        let reader_child = Arc::clone(&child);
        let maximum = self.maximum_control_bytes;
        let (sender, receiver) = mpsc::sync_channel(1);
        thread::spawn(move || {
            let response = read_frame(&mut stdout, maximum).and_then(|frame| {
                if frame.kind == FrameKind::ControlJson {
                    Ok(frame.payload)
                } else {
                    Err(ProtocolError::InvalidFrameKind(frame.kind as u8))
                }
            });
            let status = reader_child
                .lock()
                .map_err(|_| ProtocolError::PoisonedProcessLock)
                .and_then(|mut process| process.wait().map_err(ProtocolError::Io));
            let result = response.and_then(|payload| match status {
                Ok(status) if status.success() => Ok(payload),
                Ok(_) => Err(ProtocolError::WorkerFailed),
                Err(error) => Err(error),
            });
            let _ = sender.send(result);
        });

        match receiver.recv_timeout(timeout) {
            Ok(result) => result,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if let Ok(mut process) = child.lock() {
                    let _ = process.kill();
                }
                Err(ProtocolError::Timeout)
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(ProtocolError::WorkerFailed),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn frames_round_trip() {
        let mut bytes = Vec::new();
        write_frame(&mut bytes, FrameKind::PcmS16Le, b"audio", 100).unwrap();
        assert_eq!(
            read_frame(&mut Cursor::new(bytes), 100).unwrap(),
            Frame {
                kind: FrameKind::PcmS16Le,
                payload: b"audio".to_vec(),
            }
        );
    }

    #[test]
    fn declared_size_is_checked_before_allocation() {
        let mut bytes = Vec::from(MAGIC);
        bytes.push(VERSION);
        bytes.push(FrameKind::ControlJson as u8);
        bytes.extend_from_slice(&u32::MAX.to_be_bytes());
        assert!(matches!(
            read_frame(&mut Cursor::new(bytes), 1024),
            Err(ProtocolError::FrameTooLarge { .. })
        ));
    }

    #[test]
    fn writer_refuses_oversized_audio() {
        assert!(matches!(
            write_frame(&mut Vec::new(), FrameKind::PcmS16Le, &[0; 11], 10),
            Err(ProtocolError::FrameTooLarge { .. })
        ));
    }
}
