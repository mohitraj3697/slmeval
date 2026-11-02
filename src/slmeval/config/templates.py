"""Files written by `slmeval init`.

The comments in `eval.toml` are the real documentation — people read the
generated config to learn the tool long before they read the README, so every
key says what it is *for*, not just what it does.
"""

from __future__ import annotations

NAME_TOKEN = "__BENCHMARK_NAME__"

EVAL_TOML = f"""\
# slmeval benchmark definition
#
# Three commands to a scorecard:
#     slmeval doctor      check this machine can run the benchmark
#     slmeval validate    check this file and the dataset
#     slmeval run         run the matrix and print the table
#
# Nothing here contains a secret: API keys are referenced by the *name* of an
# environment variable, so this file is safe to commit.

[benchmark]
name = "{NAME_TOKEN}"

# Golden cases, one JSON object per line. Relative paths resolve against the
# directory holding this file, not the current working directory.
dataset = "./datasets/golden_cases.jsonl"

# How many times each case runs against each model. Agents are not
# deterministic — temperature 0 is not determinism on any backend — so a single
# run tells you almost nothing. 3 is the floor for computing pass^k; 5-10 gives
# you confidence intervals tight enough to gate on.
runs_per_case = 3

# record  execute real tools once and cache the results
# replay  intercept every tool call and serve it from the cache (default)
# auto    replay when cached, record when not
# live    no interception — hits your real APIs. Prints a warning.
#
# replay is what makes this runnable in CI with no GPU and no API keys.
tool_mode = "replay"

# Fraction of tool calls that get a synthetic failure (429, timeout, empty
# result, corrupted payload). What actually breaks small models in production is
# not the happy path, it is a rate limit on step 4 and no recovery. Start at 0,
# then try 0.1.
fault_injection_rate = 0.0

# Seeds the fault schedule, so every candidate meets the same failures at the
# same steps. Change it to test a different adversarial pattern.
seed = 42

# Concurrent executions. Note: local model latency measured under concurrency is
# meaningless, because requests queue inside the server. Use
# `slmeval run --serial-latency` when you want numbers you can publish.
concurrency = 4

# Which faults, and how they reach your agent. Optional: with just
# fault_injection_rate above, every fault in the taxonomy is enabled at its
# default delivery. Uncomment to narrow it.
#
# Delivery is the choice that matters. A 429 *raised* tests your retry
# wrapper; the same 429 *returned* as a payload tests whether the model reads
# its tool output at all - and that is where small models actually fail.
#
# [benchmark.faults]
# kinds = ["empty_result", "semantic_conflict", "rate_limit_429"]
# delivery = {{ rate_limit_429 = "return" }}
# weights = {{ empty_result = 3.0 }}
# delay_scale = 1.0          # 0.0 skips the real waits; fast, but no timeout fires

# Abort limits per execution. A runaway loop is a finding, not a hang.
timeout_s = 120.0
max_steps = 25


[target]
# Your agent, as "module.path:callable". It takes the `input` dict from a case
# and returns the final state. Any framework, or none — the harness supplies the
# model router and the tool sandbox by ambient context.
entrypoint = "slmeval_target:run_case"

# "generic"   you call slmeval.chat_model() in your own code and mark your
#             steps with `with slmeval.node("name")`
# "langgraph" you hand the builder to slmeval and mark nothing:
#               builder = slmeval.instrument_langgraph(StateGraph(State))
#             Needs the extra: pip install "slmeval[langgraph]"
adapter = "generic"


# The reference everything is measured against.
#
# This starts as the built-in `fake` provider so `slmeval run` works offline
# with zero setup. Point it at your real baseline when you are ready:
#
#   [baseline]
#   provider = "openai"
#   model = "gpt-4o"
#   api_key_env = "OPENAI_API_KEY"
#
# Unscripted, `fake` answers every schema with a blank but valid object - which
# runs, and passes nothing. Give it a script and the baseline becomes a free,
# reproducible reference: one turn per model call, the last turn repeating.
#
#   extra_body.script = [
#       {{ text = '{{"intent": "order_status"}}' }},
#       {{ kind = "fenced_json", text = '{{"final_answer": "shipped"}}' }},
#   ]
#
# `kind` names a misbehaviour to script instead - fenced_json, invalid_schema,
# rate_limit, timeout, hallucinated_tool, loop and the rest. A long script
# belongs in its own file: extra_body.script_file = "./fakes/baseline.toml".
[baseline]
provider = "fake"
model = "scripted-baseline"


# Candidates: the cheaper models you want to prove are good enough.
#
# `target_nodes` is the whole point of the tool. Leave it out and the candidate
# replaces the model everywhere, which almost always fails. Name the one or two
# nodes that do not need frontier reasoning — extraction, routing, formatting —
# and leave the hard reasoning on the baseline. That is where the 70% cost cut
# with no accuracy loss actually lives.
[[candidates]]
id = "gemma3-4b-local"
provider = "ollama"
model = "gemma3:4b"
target_nodes = ["intent_router"]

# A second candidate, for a model served by anything that speaks the OpenAI API
# (llama.cpp server, LM Studio, vLLM, Together, Groq, OpenRouter).
#
# [[candidates]]
# id = "qwen-7b"
# provider = "openai_compatible"
# base_url = "http://localhost:8000/v1"
# model = "Qwen/Qwen2.5-7B-Instruct-AWQ"
# api_key_env = "LOCAL_SERVER_KEY"
# target_nodes = ["intent_router"]
#
# Small models fail on strict JSON far more often than on reasoning. Pin the
# structured-output strategy to keep a comparison honest, instead of letting
# each model quietly fall back to a more forgiving one:
#   structured_output = "json_mode"


# What `slmeval gate` enforces, so CI can call it with no flags and the
# thresholds your team agreed on get reviewed with the benchmark rather than
# buried in a pipeline YAML. Every flag on the command overrides what is here.
#
# Nothing is set by default except the safety check, and that is deliberate: a
# threshold nobody chose is a threshold nobody defends the first time it fires
# spuriously, and by then the whole gate gets switched off.
#
# [gate]
# min_pass_k = 0.90          # floor on pass^k
# use_wilson_lower = true    # the default, and the point of runs_per_case:
#                            # 3 passes out of 3 is a lower bound near 44%, not
#                            # 100%, and a gate that reads the point estimate
#                            # waves through a model failing one call in four
# max_cost_ratio = 0.40      # candidate must cost <= 40% of the baseline
# max_p95_ratio = 2.0        # needs a --serial-latency run; skipped otherwise
# max_regression = 0.05      # allowed drop vs the committed slmeval.baseline.json
# no_forbidden_tools = true  # on by default; a hard fail, never a rate
#
# Exit codes are a contract: 0 pass, 1 gate failed, 2 usage, 3 provider
# unreachable, 4 cassette coverage too thin to judge. 3 and 4 exist so CI never
# reports "Ollama was not running" as "the model got worse".


# [judge]
# An LLM judge, for the cases a deterministic assertion cannot express. Off
# until you name a model here, and deliberately so: tiers 1 and 2 cost nothing,
# re-run for free and carry most of the signal.
#
# It only grades a case that carries a `judge_rubric` and already passed
# everything cheaper, it never learns which model wrote which answer, and its
# judgements are cached by content so re-rendering a report costs nothing.
#
# The one rule worth knowing before you turn it on: a judge whose agreement
# with human labels you have not measured is a random number generator with a
# confident tone. So measure it, with ~30 pairs you labelled yourself:
#
#   slmeval judge calibrate labels.jsonl   # writes slmeval.judge.json - commit it
#
# provider = "openai"
# model = "gpt-4o-mini"
# api_key_env = "OPENAI_API_KEY"
# min_agreement = 0.7        # on the Wilson lower bound, as everywhere else
"""

DATASET_JSONL = """\
{"case_id": "order_status_happy", "input": {"user_query": "Where is my order ORD-9921?"}, "tags": ["happy_path"], "assert": {"state": {"intent": "order_status", "order_id": "ORD-9921"}, "required_tools": ["fetch_order"], "forbidden_tools": ["issue_refund"], "max_tool_calls": 3, "reference_trajectory": [["intent_router", "order_lookup"]]}}
{"case_id": "order_status_messy_id", "input": {"user_query": "any update on 9921? ordered last tuesday"}, "tags": ["happy_path", "ambiguous"], "assert": {"state_contains": {"intent": "order_status"}, "forbidden_tools": ["issue_refund"], "max_tool_calls": 4}}
{"case_id": "refund_requires_confirmation", "input": {"user_query": "This is broken. Refund me now."}, "tags": ["safety"], "assert": {"state_contains": {"intent": "refund"}, "forbidden_tools": ["issue_refund"], "final_answer_regex": "(?i)(confirm|verify|check)", "max_tool_calls": 3}}
"""

TARGET_PY = '''\
"""slmeval integration point.

The contract is one coroutine. It receives the `input` dict from a golden case
and returns the agent's final state as a dict. Raise anything on failure; the
harness classifies it.

To wire up your real agent, replace the body of `run_case`. Two touch points
make it measurable:

    from slmeval import chat_model, node, sandbox_tool

    @sandbox_tool(match_on=["order_id"])
    async def fetch_order(order_id: str) -> dict:
        ...                                  # recorded once, replayed forever

    async def run_case(case_input: dict) -> dict:
        with node("intent_router"):          # names the node in the scorecard
            llm = chat_model()               # routed per candidate, per node
            decision = await llm.complete(..., schema=RoutingDecision)
        ...

Both helpers are no-ops outside a benchmark run, so this stays safe to leave in
production code.
"""

from __future__ import annotations


async def run_case(case_input: dict) -> dict:
    """Run the agent once against a single case."""
    query = case_input.get("user_query", "")

    # --- replace everything below with your agent ----------------------------
    return {
        "intent": "unknown",
        "order_id": "",
        "final_answer": f"stub target received: {query!r}",
    }
'''


def eval_toml(benchmark_name: str) -> str:
    return EVAL_TOML.replace(NAME_TOKEN, benchmark_name)


GITIGNORE_ENTRY = ".slmeval/"
