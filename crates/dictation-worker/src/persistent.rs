//! Long-lived private worker with per-request deadlines.
//!
//! Loading whisper.cpp or llama.cpp weights for every partial hypothesis would
//! miss the latency targets, so a worker process may serve many requests. Any
//! timeout, protocol violation, or unexpected exit kills the process; the next
//! request starts a fresh one. A stale response is never delivered to a newer
//! request.

use std::{
    ffi::OsString,
    io::Write,
    path::PathBuf,
    process::{Child, ChildStdin, Command, Stdio},
    sync::mpsc::{self, Receiver, RecvTimeoutError},
    thread,
    time::{Duration, Instant},
};

use crate::{
    FrameKind, ProtocolError, read_frame,
    messages::{PROTOCOL_VERSION, Ready, Request, Response},
    write_frame,
};

#[derive(Debug, Clone)]
pub struct WorkerSpec {
    pub executable: PathBuf,
    pub arguments: Vec<OsString>,
    pub working_directory: PathBuf,
    pub maximum_control_bytes: usize,
    pub maximum_audio_bytes: usize,
    pub startup_timeout: Duration,
    /// Variables passed through a cleared environment, such as thread limits.
    pub environment: Vec<(OsString, OsString)>,
}

struct Running {
    child: Child,
    stdin: ChildStdin,
    responses: Receiver<Result<Vec<u8>, ProtocolError>>,
    ready: Ready,
}

pub struct PersistentWorker {
    spec: WorkerSpec,
    running: Option<Running>,
    next_request_id: u64,
}

impl PersistentWorker {
    #[must_use]
    pub const fn new(spec: WorkerSpec) -> Self {
        Self {
            spec,
            running: None,
            next_request_id: 1,
        }
    }

    #[must_use]
    pub fn is_running(&self) -> bool {
        self.running.is_some()
    }

    /// Operating-system process identifier of the running worker.
    #[must_use]
    pub fn process_id(&self) -> Option<u32> {
        self.running.as_ref().map(|running| running.child.id())
    }

    #[must_use]
    pub fn ready_report(&self) -> Option<&Ready> {
        self.running.as_ref().map(|running| &running.ready)
    }

    /// Starts the worker if needed and waits for its ready frame.
    ///
    /// # Errors
    ///
    /// Fails when the process cannot start, does not report readiness in time,
    /// or speaks an incompatible protocol.
    pub fn ensure_started(&mut self) -> Result<&Ready, ProtocolError> {
        if self.running.is_none() {
            self.running = Some(self.spawn()?);
        }
        self.running
            .as_ref()
            .map(|running| &running.ready)
            .ok_or(ProtocolError::WorkerFailed)
    }

    /// Allocates the identifier for the next request.
    pub const fn next_request_id(&mut self) -> u64 {
        let id = self.next_request_id;
        self.next_request_id = self.next_request_id.wrapping_add(1);
        id
    }

    /// Sends one request and waits for its matching response.
    ///
    /// # Errors
    ///
    /// Returns an error on timeout, protocol violation, or worker exit. The
    /// process is killed in every error case so no half-finished request can
    /// answer a later one.
    pub fn request(
        &mut self,
        request: &Request,
        pcm_s16le: Option<&[u8]>,
        timeout: Duration,
    ) -> Result<Response, ProtocolError> {
        if request.carries_audio() != pcm_s16le.is_some() {
            return Err(ProtocolError::WorkerFailed);
        }
        let control = serde_json::to_vec(request).map_err(|_| ProtocolError::WorkerFailed)?;
        self.ensure_started()?;
        let result = self.exchange(request.request_id(), &control, pcm_s16le, timeout);
        if result.is_err() {
            self.kill();
        }
        result
    }

    /// Stops the worker and releases its model memory.
    pub fn kill(&mut self) {
        if let Some(mut running) = self.running.take() {
            let _ = running.child.kill();
            let _ = running.child.wait();
        }
    }

    fn exchange(
        &mut self,
        request_id: u64,
        control: &[u8],
        pcm: Option<&[u8]>,
        timeout: Duration,
    ) -> Result<Response, ProtocolError> {
        let deadline = Instant::now() + timeout;
        let maximum_control = self.spec.maximum_control_bytes;
        let maximum_audio = self.spec.maximum_audio_bytes;
        let running = self.running.as_mut().ok_or(ProtocolError::WorkerFailed)?;
        write_frame(
            &mut running.stdin,
            FrameKind::ControlJson,
            control,
            maximum_control,
        )?;
        if let Some(pcm) = pcm {
            write_frame(&mut running.stdin, FrameKind::PcmS16Le, pcm, maximum_audio)?;
        }
        running.stdin.flush()?;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            let payload = match running.responses.recv_timeout(remaining) {
                Ok(result) => result?,
                Err(RecvTimeoutError::Timeout) => return Err(ProtocolError::Timeout),
                Err(RecvTimeoutError::Disconnected) => return Err(ProtocolError::WorkerFailed),
            };
            let response: Response =
                serde_json::from_slice(&payload).map_err(|_| ProtocolError::WorkerFailed)?;
            if response.request_id() == request_id {
                return Ok(response);
            }
            // A response to an earlier, abandoned request: ignore it.
        }
    }

    fn spawn(&self) -> Result<Running, ProtocolError> {
        // A relative executable would resolve against the child's working
        // directory and could pick up an unintended binary.
        if !self.spec.executable.is_absolute() {
            return Err(ProtocolError::WorkerSpawn(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "worker executable path must be absolute",
            )));
        }
        let mut command = Command::new(&self.spec.executable);
        command
            .args(&self.spec.arguments)
            .current_dir(&self.spec.working_directory)
            .env_clear()
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        for (key, value) in &self.spec.environment {
            command.env(key, value);
        }
        let mut child = command.spawn().map_err(ProtocolError::WorkerSpawn)?;
        let stdin = child.stdin.take().ok_or(ProtocolError::MissingPipe)?;
        let mut stdout = child.stdout.take().ok_or(ProtocolError::MissingPipe)?;
        let maximum = self.spec.maximum_control_bytes;
        let (sender, responses) = mpsc::sync_channel(4);
        thread::spawn(move || {
            loop {
                let frame = read_frame(&mut stdout, maximum).and_then(|frame| {
                    if frame.kind == FrameKind::ControlJson {
                        Ok(frame.payload)
                    } else {
                        Err(ProtocolError::InvalidFrameKind(frame.kind as u8))
                    }
                });
                let failed = frame.is_err();
                if sender.send(frame).is_err() || failed {
                    break;
                }
            }
        });
        let ready = match responses.recv_timeout(self.spec.startup_timeout) {
            Ok(Ok(payload)) => serde_json::from_slice::<Ready>(&payload).ok(),
            _ => None,
        };
        match ready {
            Some(ready) if ready.protocol == PROTOCOL_VERSION => Ok(Running {
                child,
                stdin,
                responses,
                ready,
            }),
            other => {
                let _ = child.kill();
                let _ = child.wait();
                Err(if other.is_some() {
                    ProtocolError::UnsupportedVersion(0)
                } else {
                    ProtocolError::Timeout
                })
            }
        }
    }
}

impl Drop for PersistentWorker {
    fn drop(&mut self) {
        self.kill();
    }
}
