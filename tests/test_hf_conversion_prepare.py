"""Tokenizer preparation requires no ML dependencies or weight downloads."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("convert_hf", Path(__file__).parents[1] / "scripts/convert-hf-whisper.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_fast_tokenizer_export(tmp_path):
    (tmp_path / "tokenizer.json").write_text(json.dumps({
        "model": {"vocab": {"a": 0, "b": 1}},
        "added_tokens": [{"content": "<special>", "id": 2}],
    }))
    module.prepare_tokenizer(tmp_path)
    assert json.loads((tmp_path / "vocab.json").read_text()) == {"a": 0, "b": 1}
    assert json.loads((tmp_path / "added_tokens.json").read_text()) == {"<special>": 2}
    module.prepare_tokenizer(tmp_path)


def test_sparse_vocabulary_is_rejected(tmp_path):
    (tmp_path / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"a": 2}}}))
    with pytest.raises(ValueError, match="contiguous"):
        module.prepare_tokenizer(tmp_path)
