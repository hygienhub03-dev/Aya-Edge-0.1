"""Small JSON-schema subset used to validate tool arguments.

This intentionally does not depend on ``jsonschema``: benchmark fixtures should
be runnable in a minimal training/evaluation environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str


class SchemaValidationError(ValueError):
    def __init__(self, issues: list[ValidationIssue]):
        self.issues = issues
        super().__init__("; ".join(f"{i.path}: {i.message}" for i in issues))


def _type_ok(value: Any, expected: str) -> bool:
    return {
        "object": isinstance(value, Mapping),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(expected, True)


def validate_json(value: Any, schema: Mapping[str, Any], path: str = "$ ".strip()) -> list[ValidationIssue]:
    """Return all issues found for the supported JSON-schema subset."""
    issues: list[ValidationIssue] = []
    expected = schema.get("type")
    if expected and not _type_ok(value, expected):
        return [ValidationIssue(path, f"expected {expected}")]
    if "enum" in schema and value not in schema["enum"]:
        issues.append(ValidationIssue(path, f"must be one of {schema['enum']!r}"))
    if isinstance(value, Mapping):
        required = schema.get("required", [])
        for name in required:
            if name not in value:
                issues.append(ValidationIssue(f"{path}.{name}", "is required"))
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in properties:
                    issues.append(ValidationIssue(f"{path}.{name}", "additional property is not allowed"))
        for name, child_schema in properties.items():
            if name in value:
                issues.extend(validate_json(value[name], child_schema, f"{path}.{name}"))
    elif isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            issues.extend(validate_json(item, schema["items"], f"{path}[{index}]"))
    elif isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            issues.append(ValidationIssue(path, f"must have at least {schema['minLength']} characters"))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            issues.append(ValidationIssue(path, f"must be >= {schema['minimum']}"))
    return issues


def validate_arguments(arguments: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    """Raise :class:`SchemaValidationError` unless arguments satisfy schema."""
    issues = validate_json(arguments, schema)
    if issues:
        raise SchemaValidationError(issues)


def is_valid(arguments: Any, schema: Mapping[str, Any]) -> bool:
    return not validate_json(arguments, schema)
