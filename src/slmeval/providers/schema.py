"""Turning a Pydantic model into something each ladder rung can carry.

Each rung needs the same schema in a different shape: a tool's `parameters`, a
strict `response_format`, or plain text in the prompt. Keeping those conversions
here means a provider quirk is fixed once.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError

MAX_ERROR_LINES = 8


def json_schema_for(model: type[BaseModel]) -> dict[str, Any]:
    """The model's JSON schema, with titles stripped.

    Titles are noise that costs prompt tokens on every single call - and with a
    4B model and an 8k context, prompt tokens are the budget that runs out.
    """
    stripped = _strip_titles(model.model_json_schema())
    assert isinstance(stripped, dict)
    return stripped


def strict_schema_for(model: type[BaseModel]) -> dict[str, Any]:
    """A schema acceptable to OpenAI-style strict ``json_schema`` mode.

    Strict mode requires ``additionalProperties: false`` on every object and
    every property listed in ``required``. Optional fields are expressed as a
    nullable union instead, which is why a model with defaults still works:
    the model must emit the key, but it may emit ``null``.
    """
    strict = _strictify(json_schema_for(model))
    assert isinstance(strict, dict)
    return strict


def response_format_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "strict": True,
            "schema": strict_schema_for(model),
        },
    }


def schema_instruction(model: type[BaseModel]) -> str:
    """The prompt text used by the ``json_mode`` and ``prompt`` rungs.

    The word "JSON" is load-bearing: OpenAI's ``json_object`` mode rejects a
    request whose messages never mention it.
    """
    schema = json.dumps(json_schema_for(model), indent=2, sort_keys=True)
    return (
        "Respond with a single JSON object and nothing else - no prose, no "
        "markdown fence, no explanation. It must validate against this JSON "
        f"Schema:\n\n{schema}"
    )


def repair_instruction(model: type[BaseModel], emitted: str | None, error: str) -> str:
    """The one re-ask. Shows the model its own output and the exact failure.

    Appending only "that was invalid" produces the same invalid output again;
    showing the validation error is what makes the repair rung worth a call.
    """
    shown = (emitted or "").strip()
    if len(shown) > 1200:
        shown = shown[:1200] + "..."
    return (
        "Your previous response did not validate.\n\n"
        f"You sent:\n{shown}\n\n"
        f"The errors were:\n{error}\n\n"
        f"Send the corrected JSON object for {model.__name__} only - "
        "no prose, no fence."
    )


def validate_into(model: type[BaseModel], value: Any) -> BaseModel:
    """Validate, or raise ValidationError for the repair rung to quote."""
    return model.model_validate(value)


def format_validation_error(exc: ValidationError) -> str:
    """Pydantic's error list as something worth putting in a prompt.

    Trimmed, because a model with twenty errors will be given eight and asked
    again; dumping all twenty wastes the context we are trying to measure.
    """
    lines: list[str] = []
    for err in exc.errors()[:MAX_ERROR_LINES]:
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"- {loc}: {err['msg']}")
    extra = len(exc.errors()) - MAX_ERROR_LINES
    if extra > 0:
        lines.append(f"- ... and {extra} more")
    return "\n".join(lines)


def _strip_titles(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _strip_titles(v) for k, v in node.items() if k != "title"}
    if isinstance(node, list):
        return [_strip_titles(v) for v in node]
    return node


def _strictify(node: Any) -> Any:
    if isinstance(node, list):
        return [_strictify(v) for v in node]
    if not isinstance(node, dict):
        return node

    out = {k: _strictify(v) for k, v in node.items()}
    props = out.get("properties")
    if isinstance(props, dict):
        out["additionalProperties"] = False
        out["required"] = list(props.keys())
    return out


__all__ = [
    "format_validation_error",
    "json_schema_for",
    "repair_instruction",
    "response_format_json_schema",
    "schema_instruction",
    "strict_schema_for",
    "validate_into",
]
