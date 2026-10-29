"""The VCR tool sandbox.

Recording a tool call once and replaying it is what makes a benchmark possible
at all: you cannot run 500 trajectories against a customer's live Stripe, ERP or
production database, and you cannot compare two models against an API that
returns different data every call.

Three pieces, and they are separable on purpose:

:mod:`slmeval.sandbox.keys`
    What identifies a recorded call, and the declared per-tool policy that
    decides when two calls are the same call.
:mod:`slmeval.sandbox.normalizers`
    The named normalizers that policy refers to.
:mod:`slmeval.sandbox.envelope`
    How a return value is stored so the agent sees the same *types* on replay
    that it saw on record - and how recording refuses, loudly, when it cannot
    promise that.

:mod:`slmeval.sandbox.store`
    Where recordings live: SQLite with an LRU in front, so the second replay of
    a case never touches the disk.
:mod:`slmeval.sandbox.tool`
    ``@sandbox_tool`` and the four modes - ``record``, ``replay``, ``auto``,
    ``live``.
:mod:`slmeval.sandbox.diagnosis`
    What a miss *means*: the nearest recorded call, a per-argument diff, and a
    verdict that separates "the model got it wrong" from "we never recorded
    this path".
:mod:`slmeval.sandbox.faults`
    Deterministic fault injection, scheduled per tool per call index so one
    extra call cannot shift every later fault onto a different step.

The one invariant across all of it: no path falls back to the real tool by
accident. A replay miss is a finding and surfaces as one, because silently
calling a customer's live API instead would spend their money and invalidate
the run without telling anyone.
"""

from __future__ import annotations

from slmeval.sandbox.diagnosis import (
    ArgDiff,
    MissDiagnosis,
    Verdict,
    argument_distance,
    diagnose,
    diff_arguments,
)
from slmeval.sandbox.envelope import (
    EnvelopeDecodeError,
    UnserializableResponse,
    decode,
    decode_json,
    encode,
    encode_json,
)
from slmeval.sandbox.faults import (
    Delivery,
    Fault,
    FaultError,
    FaultKind,
    FaultSchedule,
    FaultVector,
    InjectedFault,
    ParityReport,
    RateLimited,
    ToolTimeout,
    apply_fault,
    compare_vectors,
    fault_schedule,
)
from slmeval.sandbox.keys import (
    CASCADE,
    CassetteKeys,
    KeyEntry,
    MatchPolicy,
    MatchTier,
    PolicyError,
    canonicalize,
    declare,
)
from slmeval.sandbox.normalizers import (
    Normalizer,
    UnknownNormalizer,
    resolve,
    sql_backend,
)
from slmeval.sandbox.store import Cassette, CassetteStore, Hit, StoreStats
from slmeval.sandbox.tool import (
    CassetteMiss,
    SandboxedTool,
    SandboxNotReady,
    registered_tools,
    sandbox_tool,
)

__all__ = [
    "CASCADE",
    "ArgDiff",
    "Cassette",
    "CassetteKeys",
    "CassetteMiss",
    "CassetteStore",
    "Delivery",
    "EnvelopeDecodeError",
    "Fault",
    "FaultError",
    "FaultKind",
    "FaultSchedule",
    "FaultVector",
    "Hit",
    "InjectedFault",
    "KeyEntry",
    "MatchPolicy",
    "MatchTier",
    "MissDiagnosis",
    "Normalizer",
    "ParityReport",
    "PolicyError",
    "RateLimited",
    "SandboxNotReady",
    "SandboxedTool",
    "StoreStats",
    "ToolTimeout",
    "UnknownNormalizer",
    "UnserializableResponse",
    "Verdict",
    "apply_fault",
    "argument_distance",
    "canonicalize",
    "compare_vectors",
    "declare",
    "decode",
    "decode_json",
    "diagnose",
    "diff_arguments",
    "encode",
    "encode_json",
    "fault_schedule",
    "registered_tools",
    "resolve",
    "sandbox_tool",
    "sql_backend",
]
