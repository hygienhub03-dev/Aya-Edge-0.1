"""Native agent SFT preparation for the AYA Edge causal LM.

The project model is a small custom PyTorch CausalLM rather than a
Transformers PreTrainedModel, so this module implements the same important
SFT semantics locally: structured tool-call validation, deterministic chat
serialization, and assistant-only labels using ``-100`` for ignored targets.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from aya_edge.tokenizer.bpe import SPECIAL_TOKEN_IDS


@dataclass(frozen=True)
class EncodedExample:
    """One encoded conversation with labels aligned to input_ids."""

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    uuid: str | None = None
    subset: str | None = None

    @property
    def supervised_tokens(self) -> int:
        return sum(label != -100 for label in self.labels)


class SFTDataError(ValueError):
    """Raised when a trajectory cannot be safely used for SFT."""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _function_schema(tool: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(tool, Mapping) or tool.get("type") != "function":
        raise SFTDataError("each tool must be a {type:function} JSON-schema object")
    function = tool.get("function")
    if not isinstance(function, Mapping):
        raise SFTDataError("tool.function must be an object")
    name = function.get("name")
    parameters = function.get("parameters")
    if not isinstance(name, str) or not name.strip():
        raise SFTDataError("tool function name must be a non-empty string")
    if not isinstance(parameters, Mapping):
        raise SFTDataError(f"tool {name!r} is missing an object parameters schema")
    return function


def validate_tools(tools: Any) -> dict[str, Mapping[str, Any]]:
    """Validate tool declarations and return them indexed by function name."""
    if tools is None:
        return {}
    if not isinstance(tools, list):
        raise SFTDataError("tools must be a list")
    indexed: dict[str, Mapping[str, Any]] = {}
    for tool in tools:
        function = _function_schema(tool)
        name = str(function["name"])
        if name in indexed:
            raise SFTDataError(f"duplicate tool name: {name}")
        indexed[name] = function
    return indexed


def _validate_arguments(schema: Mapping[str, Any], arguments: Any, tool_name: str) -> None:
    if not isinstance(arguments, dict):
        raise SFTDataError(f"tool {tool_name!r} arguments must be a JSON object")
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        # The repository's dev/data extras include jsonschema. Keep a safe,
        # dependency-light fallback for preflight in minimal environments.
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        missing = [key for key in required if key not in arguments]
        unknown = [key for key in arguments if isinstance(properties, Mapping) and key not in properties and schema.get("additionalProperties") is False]
        if missing or unknown:
            raise SFTDataError(f"invalid arguments for {tool_name!r}: missing={missing}, unknown={unknown}")
        return
    errors = sorted(Draft202012Validator(dict(schema)).iter_errors(arguments), key=lambda error: list(error.path))
    if errors:
        location = ".".join(str(part) for part in errors[0].path) or "<root>"
        raise SFTDataError(f"invalid arguments for {tool_name!r} at {location}: {errors[0].message}")


def _normalise_call(call: Any, tools: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if not isinstance(call, Mapping):
        raise SFTDataError("tool_calls entries must be objects")
    function = call.get("function", call)
    if not isinstance(function, Mapping):
        raise SFTDataError("tool call function must be an object")
    name = function.get("name")
    if not isinstance(name, str) or name not in tools:
        raise SFTDataError(f"tool call references undeclared tool: {name!r}")
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise SFTDataError(f"tool {name!r} arguments are not valid JSON") from exc
    _validate_arguments(tools[name].get("parameters", {}), arguments, name)
    return {
        "id": str(call.get("id", "")),
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def validate_trajectory(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate one SFT row, including tool schemas, calls, and results."""
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise SFTDataError("messages must be a non-empty list")
    tools = validate_tools(row.get("tools", []))
    pending: list[dict[str, Any]] = []
    normalised: list[dict[str, Any]] = []
    saw_assistant = False
    for index, raw in enumerate(messages):
        if not isinstance(raw, Mapping):
            raise SFTDataError(f"message {index} must be an object")
        role = raw.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise SFTDataError(f"message {index} has invalid role {role!r}")
        content = raw.get("content", "")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise SFTDataError(f"message {index} content must be a string")
        message = {"role": role, "content": content}
        if role == "assistant":
            saw_assistant = True
            calls = raw.get("tool_calls", []) or []
            if not isinstance(calls, list):
                raise SFTDataError(f"message {index} tool_calls must be a list")
            message["tool_calls"] = [_normalise_call(call, tools) for call in calls]
            pending.extend(message["tool_calls"])
        elif role == "tool":
            if not pending:
                raise SFTDataError(f"message {index} is a tool result without a pending call")
            call_id = str(raw.get("tool_call_id", ""))
            name = str(raw.get("name", ""))
            if call_id:
                match = next((call for call in pending if call["id"] == call_id), None)
            elif name:
                match = next((call for call in pending if name and call["function"]["name"] == name), None)
            else:
                # UltraData contains older trajectories whose tool results
                # carry neither an ID nor a name. In that format the only
                # unambiguous contract is FIFO pairing.
                match = pending[0]
            if match is None:
                raise SFTDataError(f"message {index} does not match a pending tool call")
            pending.remove(match)
            message["tool_call_id"] = call_id or match["id"]
            message["name"] = name or match["function"]["name"]
        normalised.append(message)
    if pending:
        raise SFTDataError("trajectory ends with an unmatched tool call")
    if not saw_assistant:
        raise SFTDataError("trajectory contains no assistant message")
    if normalised[-1]["role"] != "assistant":
        raise SFTDataError("trajectory must end with an assistant response")
    return normalised


def _message_parts(message: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return prefix, assistant-supervised body, and suffix."""
    role = str(message["role"])
    prefix = f"<|im_start|>{role}<|im_sep|>"
    body = str(message.get("content", ""))
    if role == "assistant" and message.get("tool_calls"):
        body += "<|tool_call|>" + _json(message["tool_calls"])
    elif role == "tool":
        body = "<|tool_result|>" + _json({"name": message.get("name", ""), "tool_call_id": message.get("tool_call_id", ""), "content": body})
    return prefix, body, "<|im_end|>"


def encode_trajectory(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    max_length: int,
    include_tool_schemas: bool = True,
) -> EncodedExample:
    """Serialize and encode a validated trajectory with assistant-only labels."""
    messages = validate_trajectory(row)
    ids: list[int] = [SPECIAL_TOKEN_IDS["<|bos|>"]]
    labels: list[int] = [-100]
    declared_tools = row.get("tools") or []
    if include_tool_schemas and declared_tools:
        tool_header = "<|meta_sep|><|json|>" + _json(declared_tools) + "<|im_end|>"
        tool_ids = tokenizer.encode(tool_header)
        ids.extend(tool_ids)
        labels.extend([-100] * len(tool_ids))
    for message in messages:
        prefix, body, suffix = _message_parts(message)
        prefix_ids = tokenizer.encode(prefix)
        body_ids = tokenizer.encode(body)
        suffix_ids = tokenizer.encode(suffix)
        ids.extend(prefix_ids + body_ids + suffix_ids)
        if message["role"] == "assistant":
            labels.extend([-100] * len(prefix_ids) + body_ids + suffix_ids)
        else:
            labels.extend([-100] * (len(prefix_ids) + len(body_ids) + len(suffix_ids)))
    if len(ids) > max_length:
        raise SFTDataError(f"trajectory exceeds max_length={max_length}: {len(ids)} tokens")
    if sum(label != -100 for label in labels) == 0:
        raise SFTDataError("trajectory has no assistant target tokens")
    return EncodedExample(tuple(ids), tuple(labels), row.get("uuid"), row.get("subset"))


def read_encoded_jsonl(
    path: str | Path,
    tokenizer: Any,
    *,
    max_length: int,
    include_tool_schemas: bool = True,
) -> list[EncodedExample]:
    examples: list[EncodedExample] = []
    source = Path(path)
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SFTDataError(f"invalid JSON at {source}:{line_number}") from exc
        try:
            examples.append(
                encode_trajectory(
                    row,
                    tokenizer,
                    max_length=max_length,
                    include_tool_schemas=include_tool_schemas,
                )
            )
        except SFTDataError as exc:
            raise SFTDataError(f"{source}:{line_number}: {exc}") from exc
    if not examples:
        raise SFTDataError(f"{source} contains no examples")
    return examples


def collate(examples: Sequence[EncodedExample], *, pad_token_id: int) -> tuple[Any, Any]:
    """Pad a batch, returning torch tensors only when torch is available."""
    import torch
    width = max(len(example.input_ids) for example in examples)
    input_ids = torch.full((len(examples), width), pad_token_id, dtype=torch.long)
    labels = torch.full((len(examples), width), -100, dtype=torch.long)
    for row, example in enumerate(examples):
        length = len(example.input_ids)
        input_ids[row, :length] = torch.tensor(example.input_ids, dtype=torch.long)
        labels[row, :length] = torch.tensor(example.labels, dtype=torch.long)
    return input_ids, labels


def iter_batches(examples: Sequence[EncodedExample], batch_size: int) -> Iterable[Sequence[EncodedExample]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(examples), batch_size):
        yield examples[start:start + batch_size]
