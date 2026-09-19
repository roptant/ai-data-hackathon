//! Captures two seconds in memory and reports device/configuration errors.
//! Audio is discarded, never saved or transcribed.
use cpal::traits::{DeviceTrait, HostTrait};
use dictation_platform::microphone::MicrophoneCapture;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let host = cpal::default_host();
    if let Some(device) = host.default_input_device() {
        println!("Default input: {:?}", device.description());
        println!("Default format: {:?}", device.default_input_config());
    }
    let capture = MicrophoneCapture::start_default(5)?;
    std::thread::sleep(std::time::Duration::from_secs(2));
    capture.check_runtime()?;
    let audio = capture.stop()?;
    println!("Captured {} samples at {} Hz; discarded.", audio.samples.len(), audio.sample_rate);
    if audio.samples.is_empty() { return Err("The microphone delivered no audio frames".into()); }
    Ok(())
}
