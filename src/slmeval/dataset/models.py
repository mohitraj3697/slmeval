"""Golden case schema.

The assertion block is explicit on purpose. "Did it pass?" has to be a
mechanical question or none of the downstream statistics mean anything.

``forbidden_tools`` is the highest-value field here. A 4B model that issues a
refund when asked "where is my order" is not a 2% accuracy regression, it is a
production incident, and it gets reported as one.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import ConfigDict, Field, field_validator, model_validator

from slmeval.config.models import StrictModel

CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class Assertions(StrictModel):
    state: dict[str, Any] = Field(default_factory=dict)
    """Exact equality on final state. Keys may be dotted: "order.status"."""

    state_contains: dict[str, Any] = Field(default_factory=dict)
    """Substring / membership check at a dotted path."""

    final_answer_regex: str | None = None

    required_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    """Calling any of these is a hard fail and a separate `unsafe_action` metric."""

    max_tool_calls: int | None = Field(default=None, ge=0)

    reference_trajectory: list[list[str]] = Field(default_factory=list)
    """*Acceptable* node paths, not one golden path. Agents legitimately have
    several valid routes; scoring against a single path manufactures failures.
    Trajectory distance is the minimum over this set."""

    judge_rubric: str | None = None
    """Tier 3 only. A case with no rubric never costs a judge call."""

    @field_validator("final_answer_regex")
    @classmethod
    def _check_regex(cls, v: str | None) -> str | None:
        if v is None:
            return None
        try:
            re.compile(v)
        except re.error as exc:
            raise ValueError(f"not a valid regular expression: {exc}") from exc
        return v

    @model_validator(mode="after")
    def _check_tool_lists(self) -> Assertions:
        overlap = sorted(set(self.required_tools) & set(self.forbidden_tools))
        if overlap:
            raise ValueError(
                f"tool(s) {', '.join(overlap)} are both required and forbidden - "
                "this case can never pass"
            )
        return self

    @property
    def is_empty(self) -> bool:
        """A case with no assertions cannot fail, which is almost always a mistake."""
        return not any(
            [
                self.state,
                self.state_contains,
                self.final_answer_regex,
                self.required_tools,
                self.forbidden_tools,
                self.max_tool_calls is not None,
                self.reference_trajectory,
                self.judge_rubric,
            ]
        )


class Case(StrictModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    case_id: str
    input: dict[str, Any]
    """Passed verbatim to the target entrypoint."""
    tags: list[str] = Field(default_factory=list)
    assertions: Assertions = Field(default_factory=Assertions, alias="assert")

    @field_validator("case_id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        if not CASE_ID_RE.match(v):
            raise ValueError(
                f"{v!r} is not a valid case_id: letters, digits, '.', '_' and '-' only, "
                "max 64 chars"
            )
        return v

    @field_validator("input")
    @classmethod
    def _check_input(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not v:
            raise ValueError("must not be empty - this is what gets passed to the agent")
        return v
