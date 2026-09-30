"""Exact Ollama structured-output schemas from spec 005."""

from collections.abc import Sequence


def t1_schema() -> dict:
    return {
        "type": "object",
        "properties": {"phase": {"type": "string", "enum": ["pre", "post"]}},
        "required": ["phase"],
        "additionalProperties": False,
    }


def t1p_schema() -> dict:
    return {
        "type": "object",
        "properties": {"after": {"type": "string", "enum": ["A", "B"]}},
        "required": ["after"],
        "additionalProperties": False,
    }


def t2_schema(candidates: Sequence[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "ranking": {
                "type": "array",
                "items": {"type": "string", "enum": list(candidates)},
                "minItems": 3,
                "maxItems": 3,
            }
        },
        "required": ["ranking"],
        "additionalProperties": False,
    }
