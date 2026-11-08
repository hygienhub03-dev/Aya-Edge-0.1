import json
import subprocess
import sys

import pytest

from aya_edge.tokenizer import ByteLevelTokenizer, TokenizerConfig


def test_fixed_layout_and_unicode_json_xml_round_trip():
    tokenizer = ByteLevelTokenizer()
    text = '{"message":"Héllo 世界"}<item id="7">&amp;</item>\n'
    ids = tokenizer.encode(text)
    assert tokenizer.vocab_size == 268
    assert ids == list(text.encode("utf-8"))
    assert tokenizer.decode(ids) == text


def test_special_tokens_and_boundaries_are_deterministic():
    tokenizer = ByteLevelTokenizer()
    ids = tokenizer.encode("A<|tool_call|>{}", add_bos=True, add_eos=True)
    assert ids[0] == 257
    assert ids[-1] == 258
    assert 266 in ids
    assert tokenizer.decode(ids) == "<|bos|>A<|tool_call|>{}<|eos|>"
    assert tokenizer.decode(ids, skip_special_tokens=True) == "A{}"


def test_messages_preserve_roles_and_turns():
    tokenizer = ByteLevelTokenizer()
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": '{"x": 1}'},
        {"role": "tool_result", "content": "<ok/>"},
    ]
    ids = tokenizer.encode_messages(messages)
    rendered = tokenizer.decode_messages(ids)
    assert rendered.startswith("<|bos|><|turn_start|><|system|>")
    assert "<|user|>{\"x\": 1}<|turn_end|>" in rendered
    assert rendered.endswith("<|turn_end|><|eos|>")


def test_save_load_is_exact_and_rejects_vocabulary_changes(tmp_path):
    path = tmp_path / "tokenizer.json"
    tokenizer = ByteLevelTokenizer()
    tokenizer.save(path)
    restored = ByteLevelTokenizer.load(path)
    assert restored.config.to_dict() == tokenizer.config.to_dict()
    assert restored.encode("π <|assistant|>") == tokenizer.encode("π <|assistant|>")

    data = json.loads(path.read_text())
    data["special_token_ids"]["assistant"] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="special token IDs"):
        ByteLevelTokenizer.load(path)


def test_invalid_ids_and_message_roles_fail_loudly():
    tokenizer = ByteLevelTokenizer()
    with pytest.raises(ValueError):
        tokenizer.decode([tokenizer.vocab_size])
    with pytest.raises(ValueError):
        tokenizer.encode_messages([{"role": "developer", "content": "x"}])
    with pytest.raises(ValueError):
        TokenizerConfig(special_tokens={"only": "<x>"})


def test_preparation_script_is_offline_and_deterministic(tmp_path):
    source = tmp_path / "local.jsonl"
    source.write_text('{"text":"hello"}\n{"text":"世界"}\n', encoding="utf-8")
    output = tmp_path / "tokenizer.json"
    result = subprocess.run(
        [sys.executable, "scripts/prepare_tokenizer.py", "--input", str(source), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    assert report["vocab_size"] == 268
    assert report["input_records"] == 2
    assert ByteLevelTokenizer.load(output).vocab_size == 268
