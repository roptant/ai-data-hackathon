//! Worker-side request loop shared by the model worker executables.

use std::io::{self, BufReader, BufWriter, Write};

use crate::{
    FrameKind, ProtocolError, read_frame,
    messages::{ErrorResponse, Ready, Request, Response},
    write_frame,
};

pub const MAXIMUM_CONTROL_BYTES: usize = 4 * 1024 * 1024;
/// Ten minutes plus warm-up margin of 16 kHz mono 16-bit audio.
pub const MAXIMUM_AUDIO_BYTES: usize = 16_000 * 2 * 660;

/// Handles one decoded request. Returned errors become content-free codes.
pub trait Handler {
    /// # Errors
    ///
    /// Returns a short failure code; it must not contain transcript text.
    fn handle(&mut self, request: &Request, pcm: Option<&[i16]>) -> Result<Response, String>;
}

/// Announces readiness, then serves requests until end of input or shutdown.
///
/// # Errors
///
/// Returns an error when standard I/O fails or the host violates the protocol.
pub fn serve(ready: &Ready, handler: &mut impl Handler) -> Result<(), ProtocolError> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut input = BufReader::new(stdin.lock());
    let mut output = BufWriter::new(stdout.lock());
    send(&mut output, ready)?;
    loop {
        let frame = match read_frame(&mut input, MAXIMUM_CONTROL_BYTES) {
            Ok(frame) => frame,
            Err(ProtocolError::Io(error)) if error.kind() == io::ErrorKind::UnexpectedEof => {
                return Ok(());
            }
            Err(error) => return Err(error),
        };
        if frame.kind != FrameKind::ControlJson {
            return Err(ProtocolError::InvalidFrameKind(frame.kind as u8));
        }
        let request: Request =
            serde_json::from_slice(&frame.payload).map_err(|_| ProtocolError::WorkerFailed)?;
        let pcm = if request.carries_audio() {
            let audio = read_frame(&mut input, MAXIMUM_AUDIO_BYTES)?;
            if audio.kind != FrameKind::PcmS16Le || audio.payload.len() % 2 != 0 {
                return Err(ProtocolError::InvalidFrameKind(audio.kind as u8));
            }
            Some(
                audio
                    .payload
                    .chunks_exact(2)
                    .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };
        if let Request::Shutdown { request_id } = request {
            send(&mut output, &Response::ShuttingDown { request_id })?;
            return Ok(());
        }
        let response = handler
            .handle(&request, pcm.as_deref())
            .unwrap_or_else(|code| {
                Response::Error(ErrorResponse {
                    request_id: request.request_id(),
                    code,
                })
            });
        send(&mut output, &response)?;
    }
}

fn send(output: &mut impl Write, message: &impl serde::Serialize) -> Result<(), ProtocolError> {
    let payload = serde_json::to_vec(message).map_err(|_| ProtocolError::WorkerFailed)?;
    write_frame(
        output,
        FrameKind::ControlJson,
        &payload,
        MAXIMUM_CONTROL_BYTES,
    )?;
    output.flush()?;
    Ok(())
}
