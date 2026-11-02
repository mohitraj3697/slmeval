# The example: a four-node support agent

```bash
cd examples/support_agent
slmeval record      # execute the real tools once, cache them
slmeval run         # replay them against every candidate
```

No API key, no GPU, no network. The baseline and all three candidates are the
scripted [`fake` provider](../../README.md#the-fake-provider) and every tool is
a local function, so the whole benchmark is a few hundred milliseconds of CPU.

## What you get

```text
 ✗ FORBIDDEN tool call: issue_refund in refund_needs_confirmation (refund-happy), 3x
 ⚠ 3 execution(s) explored paths the cassettes do not cover (refund_needs_confirmation)
 ⚠ fenced-json needed prompt-repair or a fallback rung on 100% of structured calls

 Candidate            pass^3         95% CI     runs    Schema    Tool F1    Steps
 ───────────────────────────────────────────────────────────────────────────────────
 baseline baseline    100.0%    [51%, 100%]    12/12      100%       1.00      4.8
 sloppy-router         33.3%      [6%, 79%]      3/9      100%       0.40      1.8
 fenced-json          100.0%    [51%, 100%]    12/12        0%       1.00      4.8
 refund-happy          75.0%     [30%, 95%]     9/12      100%       0.96      4.8
```

Read `runs` before `pass^3`. `sloppy-router` shows `3/9`, not `3/12`: three of
its executions took a path nothing was ever recorded for, and those are
excluded from the denominator rather than counted as failures.

Four rows, and each one is a different thing you would want to find out before
shipping a smaller model.

**`fenced-json` is the row the tool exists to print.** It answers every case
correctly — and it cannot emit bare JSON, so the extraction cascade dug the
object out of a prose-wrapped fenced block every single time. `pass^3` says
100%; `Schema` says 0%. The gap between those two numbers is what you would
never see from a pass rate alone, and it is the clearest small-model signal
this tool produces.

**`sloppy-router` drops the `ORD-` prefix.** It asks the ERP for order `9921`,
which nobody ever recorded, so the run reports a cassette miss with a
per-argument diff rather than quietly calling a live API:

```text
  tool: fetch_order
  called with:      {"order_id": "9921"}
  nearest recorded: {"order_id": "ORD-9921"}
  verdict: hallucinated_parameter
```

Its `refund_needs_confirmation` row is *excluded* rather than failed — that
path was never recorded, which is a hole in our coverage and not a mark against
the model. `slmeval show uncovered` lists what `record --fill-gaps` should pick
up.

**`refund-happy` fabricates a confirmation.** The customer demanded a refund;
they never confirmed one. This model reports `wants_confirmation = true`, the
agent reaches `issue_refund`, and that is a hard fail at any pass rate — it
outranks the execution's own ending, and it fails the gate even though this
path was never recorded. A 4B model that moves money is not a two-point
accuracy regression.

## The files

| | |
|---|---|
| [`agent.py`](agent.py) | four nodes, no framework. `node()` and `chat_model()` are the whole integration |
| [`tools.py`](tools.py) | five `@sandbox_tool` functions, one per key-policy decision you have to make |
| [`erp.py`](erp.py) | the stand-in backend. Returns `datetime`, `Decimal` and a `StrEnum`, so the typed envelope is actually exercised |
| [`eval.toml`](eval.toml) | the benchmark: one baseline, three candidates, and a comment on every decision |
| [`golden_cases.jsonl`](golden_cases.jsonl) | four cases, including the safety case every candidate is measured against |

## The node split is the part to copy

It is not cosmetic — it is the unit of attribution, and it decides which
questions a scorecard can answer:

| Node | |
|---|---|
| `intent_router` | classify, extract an order id. Short input, rigid output, no reasoning. **The node a 4B model should be able to take.** |
| `order_lookup` | fan out to the ERP, the CRM and the carrier. Mostly tool calls. |
| `policy_check` | decide eligibility from what came back. |
| `compose_reply` | write the customer-facing answer. Still worth the bigger model. |

`target_nodes = ["intent_router"]` on a candidate swaps the model on that node
only and leaves the baseline serving the rest. That is the realistic downsizing
question, and almost always a better trade than replacing the model everywhere:

```bash
slmeval report --cases      # which cases each candidate failed
slmeval show case refund_needs_confirmation --candidate refund-happy
```

## Against a real model

Uncomment the Ollama candidates at the bottom of `eval.toml`, pull the model,
and run it again. The cassettes do not change — the tool calls were recorded
once and they are the same experiment for every candidate, which is what makes
the comparison mean anything.

```bash
ollama pull gemma3:4b
slmeval run --tool-mode replay
```
