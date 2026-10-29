"""Named argument normalizers.

The cassette key is a hash of a tool call's arguments, so anything that changes
the bytes changes the key. Two calls that mean the same thing must therefore not
hash differently, or the harness reports a cassette miss for its own formatting
pedantry and blames the model. That is the worst failure mode in this file: in a
report, a false miss is indistinguishable from a real hallucination, and once a
user catches us doing it once they stop believing the whole scorecard.

The opposite mistake is just as bad. Normalizing aggressively hides the thing
under test: ``order_id="9921"`` where the record said ``"ORD-9921"`` is a
genuine parameter hallucination, and a normalizer that stripped non-digits would
turn that finding into a silent pass.

So normalization is **declared per tool, per argument** - never global, never
inferred. ``upper_strip`` on an order id is obviously right; the same normalizer
on a customer's free-text note destroys the signal.

Every normalizer here is **total and type-tolerant**: handed a value of a shape
it cannot process, it returns it unchanged instead of raising. That is
deliberate. A model passing ``"two"`` where the tool declares ``quantity: int``
is precisely the hallucination the cassette key should surface as a miss - if
``round:2`` raised on it instead, the harness would crash mid-benchmark on the
model's own mistake and the finding would be lost.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

Normalizer = Callable[[Any], Any]

MAX_ROUND_DIGITS: Final = 15
"""Past float precision, so a larger value is a typo rather than an intent."""


class UnknownNormalizer(ValueError):
    """A ``normalize={...}`` spec names something that does not exist.

    Raised while the tool is being declared - at import time, not mid-run - so a
    typo surfaces before anyone spends an hour of GPU time on it.
    """

    def __init__(self, spec: str, *, detail: str | None = None) -> None:
        available = ", ".join(sorted(SIMPLE_NORMALIZERS)) + ", round:N"
        super().__init__(
            f"unknown normalizer {spec!r}"
            + (f": {detail}" if detail else "")
            + f"\n  available: {available}"
        )
        self.spec = spec


# --------------------------------------------------------------------------- #
# Text
# --------------------------------------------------------------------------- #

_WS_RE = re.compile(r"\s+")


def strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def lower(value: Any) -> Any:
    return value.lower() if isinstance(value, str) else value


def upper(value: Any) -> Any:
    return value.upper() if isinstance(value, str) else value


def lower_strip(value: Any) -> Any:
    return value.strip().lower() if isinstance(value, str) else value


def upper_strip(value: Any) -> Any:
    return value.strip().upper() if isinstance(value, str) else value


def collapse_ws(value: Any) -> Any:
    """Every whitespace run becomes one space. For prose arguments that models
    re-wrap differently without changing what they say."""
    return _WS_RE.sub(" ", value).strip() if isinstance(value, str) else value


# --------------------------------------------------------------------------- #
# Numbers and containers
# --------------------------------------------------------------------------- #


def make_round(digits: int) -> Normalizer:
    """``round:N`` - fold float noise. ``0.30000000000000004`` and ``0.3`` are
    the same number to everyone except a hash function."""

    def round_to(value: Any) -> Any:
        # bool is an int subclass; rounding True to two places is nonsense.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return value
        return round(value, digits)

    round_to.__name__ = f"round_{digits}"
    return round_to


def _sort_key(item: Any) -> str:
    # Sorting mixed types directly raises; sorting by a canonical rendering does
    # not. The rendering decides order only - it is never recorded.
    return json.dumps(item, sort_keys=True, default=repr)


def sort_list(value: Any) -> Any:
    """Order-insensitive lists. A model that emits ``["b", "a"]`` where the
    record said ``["a", "b"]`` has not got it wrong if the tool treats the
    argument as a set."""
    if isinstance(value, (list, tuple)):
        return sorted(value, key=_sort_key)
    return value


def json_canonical(value: Any) -> Any:
    """Canonical rendering of a JSON blob, whether it arrives as a string or as
    a parsed object.

    Key order and indentation stop mattering, and both shapes converge on the
    same text - so one side recording a dict while the other emits the
    equivalent string is no longer a miss.
    """
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return value
    elif isinstance(value, (Mapping, list, tuple)):
        parsed = value
    else:
        return value
    try:
        return json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return value


# --------------------------------------------------------------------------- #
# SQL
# --------------------------------------------------------------------------- #
#
# `SELECT * FROM orders WHERE id = 'X'` and `select  *  from orders where
# id='X'` are the same query. Hashing them differently is our bug, not the
# model's.
#
# `sqlglot` parses properly and is used when it is importable, but it is not a
# dependency: a `pip install slmeval` that drags in a SQL parser for a feature
# most users never touch is the worse trade. So the fallback below is the path
# actually exercised on a default install, and it is written to be conservative
# rather than clever.
#
# The one thing it must never do is fold the case of a string literal.
# `id = 'X'` and `id = 'x'` select different rows in most databases, so folding
# them would make two genuinely different calls collide - the silent pass this
# whole module exists to prevent.

_SQL_SKIP_RE = re.compile(
    r"""(?P<lit>'(?:[^']|'')*'|"(?:[^"]|"")*")|--[^\n]*|/\*.*?\*/""",
    re.DOTALL,
)

_SQL_TOKEN_RE = re.compile(
    r"""[A-Za-z_@\#][A-Za-z0-9_$]*    # identifiers, keywords, bind parameters
      | \d+(?:\.\d+)?                 # numbers
      | <=|>=|<>|!=|\|\||::|->>|->    # multi-character operators
      | [^\sA-Za-z0-9_]               # every other symbol, one token each
    """,
    re.VERBOSE,
)

_SQLGLOT: Any = None
_BACKEND: str | None = None


def sql_backend() -> str:
    """``"sqlglot"`` or ``"fallback"``.

    Worth reporting alongside a miss diagnosis: "these two queries differ" means
    something different depending on which of the two said it.
    """
    global _SQLGLOT, _BACKEND
    if _BACKEND is None:
        try:
            import sqlglot  # type: ignore[import-not-found]
        except ImportError:
            _SQLGLOT, _BACKEND = None, "fallback"
        else:
            _SQLGLOT, _BACKEND = sqlglot, "sqlglot"
    return _BACKEND


def _sql_tokens(text: str) -> list[str]:
    return _SQL_TOKEN_RE.findall(text.lower())


def sql_fallback(value: Any) -> Any:
    """Whitespace- and case-normalize SQL without parsing it.

    Comments are dropped, keywords and bare identifiers fold to lower case,
    quoted literals and quoted identifiers survive byte for byte, and every
    symbol becomes its own token - so spacing around ``=``, ``,`` and
    parentheses stops affecting the hash.
    """
    if not isinstance(value, str):
        return value
    tokens: list[str] = []
    pos = 0
    for match in _SQL_SKIP_RE.finditer(value):
        tokens.extend(_sql_tokens(value[pos : match.start()]))
        literal = match.group("lit")
        if literal is not None:
            tokens.append(literal)
        pos = match.end()
    tokens.extend(_sql_tokens(value[pos:]))
    while tokens and tokens[-1] == ";":
        tokens.pop()
    return " ".join(tokens)


def sql(value: Any) -> Any:
    """Normalize SQL, via ``sqlglot`` when it is installed.

    Any parse failure falls through to :func:`sql_fallback`: a query the parser
    rejects still has to be hashed, and an unparseable argument may itself be
    the finding.
    """
    if not isinstance(value, str):
        return value
    module = None if sql_backend() == "fallback" else _SQLGLOT
    if module is not None:
        try:
            return str(module.parse_one(value).sql(normalize=True, comments=False))
        except Exception:
            pass
    return sql_fallback(value)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

SIMPLE_NORMALIZERS: Final[Mapping[str, Normalizer]] = {
    "collapse_ws": collapse_ws,
    "json_canonical": json_canonical,
    "lower": lower,
    "lower_strip": lower_strip,
    "sort_list": sort_list,
    "sql": sql,
    "strip": strip,
    "upper": upper,
    "upper_strip": upper_strip,
}


def resolve(spec: str) -> Normalizer:
    """Turn one ``normalize={...}`` value into a callable.

    Accepts a bare name (``"upper_strip"``) or a parameterized one
    (``"round:2"``).
    """
    if not isinstance(spec, str):
        raise UnknownNormalizer(repr(spec), detail="normalizers are named by string")
    name, sep, argument = spec.partition(":")
    name = name.strip()
    if name == "round":
        if not sep:
            raise UnknownNormalizer(spec, detail="round needs a digit count, as in 'round:2'")
        try:
            digits = int(argument)
        except ValueError:
            raise UnknownNormalizer(
                spec, detail=f"{argument!r} is not an integer digit count"
            ) from None
        if not 0 <= digits <= MAX_ROUND_DIGITS:
            raise UnknownNormalizer(
                spec, detail=f"digit count must be between 0 and {MAX_ROUND_DIGITS}"
            )
        return make_round(digits)
    if sep:
        raise UnknownNormalizer(spec, detail=f"{name!r} takes no argument")
    try:
        return SIMPLE_NORMALIZERS[name]
    except KeyError:
        raise UnknownNormalizer(spec) from None


def resolve_all(specs: Mapping[str, str]) -> dict[str, Normalizer]:
    """Resolve a whole ``normalize={...}`` mapping, naming the argument on error."""
    resolved: dict[str, Normalizer] = {}
    for argument, spec in specs.items():
        try:
            resolved[argument] = resolve(spec)
        except UnknownNormalizer as exc:
            raise UnknownNormalizer(spec, detail=f"declared for argument {argument!r}") from exc
    return resolved


def apply(value: Any, normalizers: Sequence[Normalizer]) -> Any:
    for normalizer in normalizers:
        value = normalizer(value)
    return value


__all__ = [
    "MAX_ROUND_DIGITS",
    "SIMPLE_NORMALIZERS",
    "Normalizer",
    "UnknownNormalizer",
    "apply",
    "collapse_ws",
    "json_canonical",
    "lower",
    "lower_strip",
    "make_round",
    "resolve",
    "resolve_all",
    "sort_list",
    "sql",
    "sql_backend",
    "sql_fallback",
    "strip",
    "upper",
    "upper_strip",
]
