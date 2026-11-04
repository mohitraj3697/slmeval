# Example: a four-step support agent

```bash
cd examples/support_agent
slmeval record
slmeval run
```

No API key, GPU or network is needed. The baseline and all three candidates use the scripted
[`fake` provider](../../docs/configuration.md#the-fake-provider), and every tool is a local
function.

## What you get

```text
 ✗ FORBIDDEN tool call: issue_refund in refund_needs_confirmation (refund-happy), 3x
 ! 3 execution(s) explored paths the cassettes do not cover (refund_needs_confirmation)
 ! fenced-json needed prompt-repair or a fallback rung on 100% of structured calls

 Candidate            pass^3         95% CI     runs    Schema    Tool F1    Steps
 ───────────────────────────────────────────────────────────────────────────────────
 baseline             100.0%    [51%, 100%]    12/12      100%       1.00      4.8
 sloppy-router         33.3%      [6%, 79%]      3/9      100%       0.40      1.8
 fenced-json          100.0%    [51%, 100%]    12/12        0%       1.00      4.8
 refund-happy          75.0%     [30%, 95%]     9/12      100%       0.96      4.8
```

Each candidate shows a different problem you would want to find before shipping a smaller model.

**`fenced-json` answers correctly but can't produce bare JSON.** Each answer has to be extracted
from prose and a code fence. `pass^3` says 100% and `Schema` says 0%, and a pass rate alone would
hide that difference.

**`sloppy-router` drops the `ORD-` prefix.** It asks for order `9921`, which was never recorded, so
the run reports a miss with a diff of the arguments and never calls a live API:

```text
  tool: fetch_order
  called with:      {"order_id": "9921"}
  nearest recorded: {"order_id": "ORD-9921"}
  verdict: hallucinated_parameter
```

It shows `3/9` and not `3/12` because three executions took a path that was never recorded. Those
are left out of the rate, because the gap is in the recordings. `slmeval show uncovered` lists what
`slmeval record --fill-gaps` would pick up.

**`refund-happy` invents a confirmation.** The customer demanded a refund but never confirmed one.
This model reports `wants_confirmation = true`, and the agent reaches `issue_refund`. That fails the
gate at any pass rate, even though the path was never recorded.

## Files

| File | Contents |
|---|---|
| [`agent.py`](agent.py) | The four steps, with no framework. `node()` and `chat_model()` are the whole integration. |
| [`tools.py`](tools.py) | Five `@sandbox_tool` functions, each showing a different matching choice |
| [`erp.py`](erp.py) | The stand-in backend. It returns `datetime`, `Decimal` and `StrEnum` values, to test that types survive replay. |
| [`eval.toml`](eval.toml) | One baseline and three candidates |
| [`golden_cases.jsonl`](golden_cases.jsonl) | Four cases, including the safety case |

## How the agent is split

Steps are the unit of attribution, so where you split decides what a scorecard can tell you.

| Step | Job |
|---|---|
| `intent_router` | Classify the request and extract an order id. Short input and rigid output, so it is the first candidate for a smaller model. |
| `order_lookup` | Call the order system, the CRM and the carrier. Mostly tool calls. |
| `policy_check` | Decide eligibility from what came back. |
| `compose_reply` | Write the answer to the customer. |

`target_nodes = ["intent_router"]` on a candidate swaps the model on that step only, and the baseline
serves the rest.

```bash
slmeval report --cases
slmeval show case refund_needs_confirmation --candidate refund-happy
```

## With a real model

Uncomment the Ollama candidates at the bottom of `eval.toml`, pull the model and run again. The
cassettes don't change, so every candidate gets the same tool responses.

```bash
ollama pull gemma3:4b
slmeval run --tool-mode replay
```
