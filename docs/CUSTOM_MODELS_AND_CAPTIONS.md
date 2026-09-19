# Custom speech models and Live Caption

## Recording shortcut

The default `Ctrl+Alt+Space` binding is **hold to record**: releasing it stops
recording. `Ctrl+Alt+Period` is **toggle**: press once to start and again to stop.
The app itself stays open. In Setup > Shortcuts, click **Use … as start/stop
toggle** to exchange these two bindings. Existing user bindings are preserved
until you change them. The Dictation page shows the registered bindings and
explains their behavior.

## Load your own model

In Setup > Custom speech model, click **Browse**, choose a whisper.cpp GGML
`.bin` file, and click **Import / download and use**. The app copies it to its
model directory, fingerprints it, persists the selection, and starts the ASR
worker. You can switch back to a built-in model using its **Use** button.
One custom model is selectable at a time; importing another replaces that
selection. Old content-addressed weight files are retained.

You can also paste a direct HTTPS URL to GGML weights and the publisher's
SHA-256. A Hugging Face repository page is not a direct weights URL. Local
files do not require a supplied checksum. Invalid formats, checksum failures,
and worker-load errors are displayed in Setup. Installation success means the
copy passed format/checksum checks; **Speech recognition is loaded** confirms
the inference worker successfully loaded it.

The ASR runtime is whisper.cpp. It cannot directly load Transformers
`model.safetensors`, PyTorch checkpoints, adapter-only repositories, or GGUF
language models. Custom weights use ordinary Whisper timestamps without
assuming the architecture of the built-in base model.

### Your example: Grenmango/whisper-medium-en-vi-hqtv-personalized

The repository inspected on 2026-09-19 contains `model.safetensors`,
`config.json`, and `tokenizer.json`, with no GGML model. Download the complete
repository locally, then convert it once. The helper below reconstructs the
legacy vocabulary files required by the upstream converter from tokenizer.json.
It writes those small files into the downloaded model directory.

Run these commands from this project in a Python environment with sufficient
disk space and RAM for the medium checkpoint:

```powershell
python -m pip install torch "transformers<5" safetensors numpy huggingface_hub
hf download Grenmango/whisper-medium-en-vi-hqtv-personalized --local-dir .local-dictation/hf-personalized
git clone https://github.com/ggml-org/whisper.cpp .local-dictation/whisper.cpp
git clone https://github.com/openai/whisper .local-dictation/whisper
python scripts/convert-hf-whisper.py --model-dir .local-dictation/hf-personalized --whisper-cpp .local-dictation/whisper.cpp --whisper-repo .local-dictation/whisper --output .local-dictation/converted
```

Browse to `.local-dictation/converted/ggml-model.bin` in the app. For Vietnamese,
set Recognition > Language to `vi`, or leave it empty for language detection.
The default `en` forces English recognition. Full conversion and transcription
of this particular medium model have not been validated here.

Upstream converter: https://github.com/ggml-org/whisper.cpp/blob/master/models/convert-h5-to-ggml.py

## Connect the existing Live Caption application

The app at `D:\Hoai Anh\Aalto\Hobbies\live caption\live-caption-main` is already
a native HTTP/WebSocket client. It receives this project's transcripts; it
does not supply an independent speech-recognition service.

1. Open Local Dictation > Integrations and enable the local API. Confirm the
   listening port (default `8765`). Secure storage must be available.
2. From this project's directory, run:

   ```powershell
   cargo run -p dictation-api --example caption_client -- pair --port 8765 --name "Live Caption" --scopes status:read,transcript:live,transcript:final
   ```

3. In Integrations, click **Refresh pairing requests**, compare the code, and
   approve the requested caption scopes. The command prints the token once.
4. Open `live-caption-main\dist\LocalDictationCaptions.exe`. Choose
   **Configure Real API…**, enter `http://127.0.0.1:8765` and the token, then
   **Save and connect**. Choose **Real API Mode** and **Start captions**.
5. Start dictation and speak. Partials replace earlier revisions; final text
   replaces all partials. The caption app saves its token in Windows Credential
   Manager. No session-control scope is needed for displaying captions.

The Rust status endpoint now includes `version`, token-scoped `capabilities`,
and the nested `state` expected by this caption client, while keeping the
existing Rust fields. Previously the client rejected the response before
opening the event stream. The caption repository's older `dictation api pair`
instructions refer to the Python reference host; use the command above for
the Rust desktop app.
