# Choosing and fetching the two models

Written for: whoever makes the model decision the plan leaves open.

There are exactly two learned roles and no third one appears silently
(plan section 3). Both ship as **unresolved candidates**: the registry knows
their names and why they are candidates, and nothing else. No URL, no checksum,
no licence. Every component that needs a model refuses until you fill those in.

```
$ dictation models list
asr:
  whisper-base-q5_1   asr   unresolved (source_url, sha256, size_bytes, license_id, revision)
      Plan section 4: start ASR evaluation with quantized Whisper base and small...
  whisper-small-q5_1  asr   unresolved (...)
privacy:
  qwen3-4b-instruct-2507-q4  privacy  unresolved (...)
      Plan section 4: a candidate, not a claim of adequate privacy recall...
```

## The decision to make

For the ASR role, plan section 4 says: start with quantized Whisper base and
small, and choose the *smallest* model that meets measured recognition and
latency requirements for English. Benchmark on the 8 GB CPU-only baseline, not
just the 16 GB development machine.

For the privacy role, the candidate is a four-bit Qwen3-4B-Instruct-2507 through
llama.cpp. The plan is explicit that this is a candidate and not a claim of
adequate recall or acceptable speed on every 8 GB machine.

Before recording a URL, the plan requires validating:

1. the licence of the **original** model, and of the converted artifact, which
   can differ,
2. who produced the quantized artifact and from which upstream revision,
3. the conversion and the quantization quality,
4. redistribution terms, if you intend to bundle rather than download.

## Recording a choice

```bash
dictation models resolve whisper-base-q5_1 \
  --url      https://<exact artifact URL> \
  --sha256   <64 hex characters> \
  --size     <exact byte size> \
  --license  <SPDX identifier> \
  --revision <upstream revision or commit> \
  --license-url https://... \
  --provenance "who converted and quantized this"

dictation models choose --role asr whisper-base-q5_1
dictation models fetch  --role asr
dictation models verify
```

A model that is not in the registry can be added the same way: `resolve` an
identifier that does not exist yet fails, so add it through
`ModelRegistry.upsert` (or a small script) and then resolve it.

## What the fetcher enforces

- **HTTPS only.** An `http://` URL is refused.
- **A pinned digest and size.** Both are required; an oversized response is cut
  off mid-download and refused rather than buffered.
- **Atomic install.** The download lands in `<name>.part` and is renamed only
  after the size and SHA-256 both match, so an interrupted download can never be
  loaded.
- **An allow-listed extension.** `.gguf` and `.bin` only. No archives, so there
  is no extraction step, and nothing downloaded is ever executed.
- **A manifest entry**, re-verified at startup: an artifact that changed on disk
  is refused rather than loaded (supply-chain tampering, a partially restored
  backup).

## Runtime binaries

The model files are not enough: whisper.cpp and llama.cpp binaries have to be
present. A release bundles them, so a normal user needs no Python, compiler or
model-serving application (plan section 3). In development:

```bash
export DICTATION_WHISPER_BINARY=/path/to/whisper-cli
export DICTATION_LLAMA_BINARY=/path/to/llama-cli
```

Both workers report an actionable `ModelNotInstalled` when a binary or a model
is missing, and the privacy worker runs the subprocess with a minimal
environment, a grammar file and the prompt - no tools, no network, no
credentials.

## After fetching

```bash
dictation bench          # measure both roles on this machine
dictation doctor         # what is chosen, installed and still missing
```

`bench` writes `metadata/benchmark.json` with the exact OS, CPU, memory, model,
quantization and timings, and lists which provisional targets were not met. Peak
memory is read from the operating system rather than derived from the weight
file size.

Two things stay true after a model is installed:

- **A benchmark is not a privacy evaluation.** Measured recall on the curated
  high-risk span benchmark is a separate exercise; see
  [EVALUATION.md](EVALUATION.md).
- **`UPLOAD_GATES_MET` is still `False`.** Installing a model does not enable
  automatic upload. Someone has to run the evaluation, record the numbers, and
  change that flag deliberately.
