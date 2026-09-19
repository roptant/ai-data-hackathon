#!/usr/bin/env python3
"""Prepare a downloaded HF Whisper checkpoint for the upstream GGML converter.

Requires torch, transformers, safetensors, numpy, and local clones of
ggml-org/whisper.cpp and openai/whisper. Never downloads model weights or runs
model-provided Python code. See docs/CUSTOM_MODELS_AND_CAPTIONS.md.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def prepare_tokenizer(model: Path) -> None:
    """Reconstruct legacy tokenizer files from a fast tokenizer export."""
    if (model / "vocab.json").exists():
        if not (model / "added_tokens.json").exists():
            (model / "added_tokens.json").write_text("{}", encoding="utf-8")
        return
    tokenizer_path = model / "tokenizer.json"
    if not tokenizer_path.exists():
        raise ValueError("Download the tokenizer files along with the model weights.")
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    vocab = tokenizer.get("model", {}).get("vocab")
    if not isinstance(vocab, dict) or not vocab or not all(isinstance(v, int) for v in vocab.values()):
        raise ValueError("Expected a Whisper BPE vocabulary in tokenizer.json.")
    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError("Vocabulary IDs must be contiguous for whisper.cpp conversion.")
    added = {token["content"]: token["id"] for token in tokenizer.get("added_tokens", []) if token["content"] not in vocab}
    (model / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    (model / "added_tokens.json").write_text(json.dumps(added, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--whisper-cpp", required=True, type=Path)
    parser.add_argument("--whisper-repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    model = args.model_dir.resolve()
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "whisper":
        parser.error("The checkpoint must be a Whisper model.")
    if config.get("auto_map"):
        parser.error("Custom model Python code is not supported.")
    converter = args.whisper_cpp.resolve() / "models" / "convert-h5-to-ggml.py"
    assets = args.whisper_repo.resolve() / "whisper" / "assets" / "mel_filters.npz"
    if not converter.is_file() or not assets.is_file():
        parser.error("Provide local whisper.cpp and OpenAI whisper repository directories.")
    prepare_tokenizer(model)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "ggml-model.bin").exists():
        parser.error("Output already contains ggml-model.bin; choose an empty output directory.")
    subprocess.run([sys.executable, str(converter), str(model), str(args.whisper_repo.resolve()), str(output)], check=True)
    print(f"In Local Dictation, open Setup > Custom speech model > Browse and select {output / 'ggml-model.bin'}")


if __name__ == "__main__":
    main()
