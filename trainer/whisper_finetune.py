#!/usr/bin/env python3
"""Reference personalization trainer for the Local Dictation server (plan §10).

Contract (see crates/dictation-server/src/training.rs):

    whisper_finetune.py [--hf-model ID] [--steps N] [--export-only]
        --job-dir DIR --base-model GGML --output DIR

* ``DIR/train`` and ``DIR/heldout`` hold ``*.wav`` (16 kHz mono PCM) and
  ``*.txt`` pairs of one customer only. Held-out data is never read here; the
  server evaluates the candidate itself with the desktop runtime.
* Adaptation: the encoder is frozen and the decoder fine-tuned, so the base
  model is unchanged and the customer's model is a separate artifact.
* Export: whisper.cpp ggml, f16 tensors at ``OUTPUT/model-f16.bin``. The
  server then uses the configured whisper.cpp quantizer to produce and verify
  the q5_1 ``OUTPUT/model.bin`` candidate. The header, mel filters, and
  vocabulary are copied byte-for-byte from ``--base-model`` so tokenization
  cannot drift from the runtime the desktop ships; only tensors change.
* ``--export-only`` converts the untouched Hugging Face weights, which must
  reproduce the base model's transcripts; that proves the export path.

Nothing here logs transcript text. The job directory is deleted by the server
afterwards.
"""

from __future__ import annotations

import argparse
import json
import struct
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

GGML_MAGIC = 0x67676D6C

HF_TO_GGML = {
    "self_attn.k_proj": "attn.key",
    "self_attn.q_proj": "attn.query",
    "self_attn.v_proj": "attn.value",
    "self_attn.out_proj": "attn.out",
    "self_attn_layer_norm": "attn_ln",
    "encoder_attn.q_proj": "cross_attn.query",
    "encoder_attn.k_proj": "cross_attn.key",
    "encoder_attn.v_proj": "cross_attn.value",
    "encoder_attn.out_proj": "cross_attn.out",
    "encoder_attn_layer_norm": "cross_attn_ln",
    "fc1": "mlp.0",
    "fc2": "mlp.2",
    "final_layer_norm": "mlp_ln",
    "encoder.layer_norm.bias": "encoder.ln_post.bias",
    "encoder.layer_norm.weight": "encoder.ln_post.weight",
    "encoder.embed_positions.weight": "encoder.positional_embedding",
    "decoder.layer_norm.bias": "decoder.ln.bias",
    "decoder.layer_norm.weight": "decoder.ln.weight",
    "decoder.embed_positions.weight": "decoder.positional_embedding",
    "decoder.embed_tokens.weight": "decoder.token_embedding.weight",
}

F32_TENSORS = {
    "encoder.conv1.bias",
    "encoder.conv2.bias",
    "encoder.positional_embedding",
    "decoder.positional_embedding",
}


def ggml_name(name: str) -> str | None:
    """Maps a Hugging Face parameter name to the whisper.cpp tensor name."""
    if name == "proj_out.weight":
        return None  # tied to the token embedding
    parts = name.split(".")[1:]  # drop "model."
    if len(parts) > 1 and parts[1] == "layers":
        parts[1] = "blocks"
        mapped = HF_TO_GGML[".".join(parts[3:-1])]
        return ".".join(parts[:3] + [mapped] + parts[-1:])
    joined = ".".join(parts)
    return HF_TO_GGML.get(joined, joined)


def base_header(path: Path) -> tuple[list[int], bytes]:
    """Returns the 11 hyperparameters (without ftype) and the raw mel-filter
    and vocabulary sections of a whisper.cpp model."""
    data = path.read_bytes()
    values = struct.unpack_from("<12i", data, 0)
    if values[0] != GGML_MAGIC:
        raise SystemExit("base model is not a whisper.cpp ggml file")
    offset = 4 * 12
    n_mel, n_fft = struct.unpack_from("<ii", data, offset)
    offset += 8 + 4 * n_mel * n_fft
    (n_vocab,) = struct.unpack_from("<i", data, offset)
    offset += 4
    for _ in range(n_vocab):
        (length,) = struct.unpack_from("<i", data, offset)
        offset += 4 + length
    return list(values[1:11]), data[4 * 12 : offset]


def export_ggml(model: WhisperForConditionalGeneration, base: Path, target: Path) -> None:
    hparams, sections = base_header(base)
    with target.open("wb") as out:
        out.write(struct.pack("<i", GGML_MAGIC))
        out.write(struct.pack("<10i", *hparams))
        out.write(struct.pack("<i", 1))  # ftype: f16
        out.write(sections)
        for name, tensor in model.state_dict().items():
            mapped = ggml_name(name)
            if mapped is None:
                continue
            array = tensor.detach().cpu().float().numpy().squeeze()
            if mapped in ("encoder.conv1.bias", "encoder.conv2.bias"):
                array = array.reshape(array.shape[0], 1)
            dims = array.ndim
            f16 = dims >= 2 and mapped not in F32_TENSORS
            array = array.astype(np.float16 if f16 else np.float32)
            encoded = mapped.encode()
            out.write(struct.pack("<iii", dims, len(encoded), 1 if f16 else 0))
            for index in range(dims):
                out.write(struct.pack("<i", array.shape[dims - 1 - index]))
            out.write(encoded)
            array.tofile(out)


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != 16_000 or handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise SystemExit("training audio must be 16 kHz mono 16-bit PCM")
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def load_pairs(directory: Path) -> list[tuple[np.ndarray, str]]:
    pairs = []
    for wav in sorted(directory.glob("*.wav")):
        text = wav.with_suffix(".txt").read_text(encoding="utf-8").strip()
        if text:
            pairs.append((read_wav(wav), text))
    return pairs


def fine_tune(model, processor, pairs, steps: int, learning_rate: float) -> list[float]:
    for parameter in model.model.encoder.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    model.train()
    losses = []
    prefix = processor.tokenizer.prefix_tokens
    for step in range(steps):
        audio, text = pairs[step % len(pairs)]
        features = processor.feature_extractor(audio, sampling_rate=16_000, return_tensors="pt").input_features
        labels = processor.tokenizer(text, return_tensors="pt").input_ids
        # The tokenizer already adds the start/language/task prefix.
        assert labels[0, : len(prefix)].tolist() == prefix
        output = model(input_features=features, labels=labels)
        output.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(float(output.loss))
    model.eval()
    return losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf-model", default="openai/whisper-base")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--export-only", action="store_true")
    arguments = parser.parse_args()

    torch.manual_seed(0)
    processor = WhisperProcessor.from_pretrained(arguments.hf_model, language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(arguments.hf_model, torch_dtype=torch.float32)
    report: dict[str, object] = {"hf_model": arguments.hf_model, "method": "decoder_fine_tune_frozen_encoder"}
    if not arguments.export_only:
        pairs = load_pairs(arguments.job_dir / "train")
        if not pairs:
            raise SystemExit("no training pairs")
        losses = fine_tune(model, processor, pairs, arguments.steps, arguments.learning_rate)
        report.update({"train_examples": len(pairs), "steps": arguments.steps, "first_loss": losses[0], "last_loss": losses[-1]})
    arguments.output.mkdir(parents=True, exist_ok=True)
    export_ggml(model, arguments.base_model, arguments.output / "model-f16.bin")
    (arguments.output / "train_report.json").write_text(json.dumps(report), encoding="utf-8")


if __name__ == "__main__":
    main()
