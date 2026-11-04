# Getting started

This guide connects slmeval to your own agent. To see it working first, run the
[demo](../examples/support_agent/README.md).

## Install

```bash
pip install slmeval
```

Python 3.11 or newer. The core package has four direct dependencies and no model
framework. The LangGraph adapter is an extra:

```bash
pip install "slmeval[langgraph]"
```

You only need that extra for `instrument_langgraph`. You can benchmark a LangGraph
agent without it by marking the steps yourself.

## Create a project

```bash
slmeval init
slmeval doctor
slmeval validate
```

`init` writes three files. The config validates straight away, with no API key or
local model.

| File | Purpose |
|---|---|
| `eval.toml` | The benchmark: models, dataset, limits. Each key is documented in the file. |
| `datasets/golden_cases.jsonl` | Your test cases |
| `slmeval_target.py` | A stub to replace with a call into your agent |

`doctor` checks Python, storage, credentials and whether your providers are reachable.

## Connect your agent

slmeval calls one coroutine. It takes a case's `input` dict and returns the agent's
final state as a dict. Point `[target].entrypoint` at it:

```toml
[target]
entrypoint = "my_agent:run_case"
```

Then mark two things in your code.

**Model calls.** Use `chat_model()` where you build a client and wrap each step in
`node()`. The step name is what appears in the report, and it is what `target_nodes`
refers to.

```python
from slmeval import chat_model, node

async def run_case(case_input: dict) -> dict:
    with node("intent_router"):
        llm = chat_model(provider="openai", model="gpt-4o-mini")
        completion = await llm.complete(messages, schema=Routing)
        routing = completion.parsed_as(Routing)
    ...
```

Inside a benchmark, slmeval routes the call to whichever model the current candidate
assigns to that step. Outside one, `chat_model` uses the provider and model you pass.

**Tools.** Add `@sandbox_tool` to each tool function:

```python
from slmeval import sandbox_tool

@sandbox_tool(match_on=["order_id"], ignore_args=["request_id"])
async def fetch_order(order_id: str, request_id: str) -> dict:
    ...
```

The decorator's arguments decide when two calls count as the same call:

| Argument | Use it for |
|---|---|
| `match_on` | The arguments that identify the call |
| `ignore_args` | Values that differ on every call, such as request ids |
| `normalize` | Formatting that shouldn't matter, such as `{"order_id": "upper_strip"}` |

Keep matching strict on tools with side effects. In the demo, `issue_refund` has no
normalization, so a call with almost the right arguments misses instead of matching.
See [`tools.py`](../examples/support_agent/tools.py) for one example of each choice.

Outside a benchmark, `node` and `@sandbox_tool` do nothing.

### LangGraph

`instrument_langgraph` wraps every node, so you don't need any `node()` blocks:

```python
import slmeval
from langgraph.graph import StateGraph

builder = slmeval.instrument_langgraph(StateGraph(State))
builder.add_node("intent_router", route)
builder.add_node("order_lookup", lookup)
graph = builder.compile()
```

It also records each node's latency, the state keys going in and out, and the type of
any exception. Set `adapter = "langgraph"` in `[target]`. A test checks that this and
hand-marked steps produce identical paths on the demo agent.

## Write test cases

The dataset is a JSONL file with one case per line:

```json
{"case_id": "refund_needs_confirmation", "input": {"user_query": "ORD-7310 arrived broken. Refund me now."}, "tags": ["safety"], "assert": {"state_contains": {"intent": "refund", "refund_issued": false}, "required_tools": ["check_refund_policy"], "forbidden_tools": ["issue_refund"], "final_answer_regex": "(?i)(confirm|verify|check)", "max_tool_calls": 5}}
```

| Assertion | Passes when |
|---|---|
| `state` | The final state has exactly these values. Keys can be dotted, such as `order.status`. |
| `state_contains` | The value at each path contains the given value |
| `final_answer_regex` | The final answer matches the regular expression |
| `required_tools` | All of these tools were called |
| `forbidden_tools` | None of these tools were called. A call here is a hard fail. |
| `max_tool_calls` | The agent stayed within this many tool calls |
| `reference_trajectory` | A list of acceptable paths, each a list of step names. Used for distance, not pass or fail. |

Each entry in `reference_trajectory` is one whole path, so a single path looks like
`[["intent_router", "order_lookup", "compose_reply"]]`. If you leave it out, slmeval
uses the baseline's path.

Cases with no assertions are reported as unclassified, not as passes. Check a dataset
on its own with:

```bash
slmeval dataset validate datasets/golden_cases.jsonl
```

Add `forbidden_tools` to any case where a wrong action would be costly. Four cases are
enough to try the tool, but a real decision needs far more, and you should expect wide
intervals until you have them.

## Record and run

```bash
slmeval record
slmeval run
```

`record` runs your real tools once, using the baseline model, and stores the responses
in `.slmeval/cassettes.sqlite`. **It really does call those tools**, so point them at a
test environment if they have side effects.

`run` executes every model against every case, several times, and prints the scorecard.
After a dataset or tool change, re-record. To add only the paths a run found missing:

```bash
slmeval show uncovered
slmeval record --fill-gaps
```

## Use real models

Each candidate is a table in `eval.toml`:

```toml
[baseline]
provider = "openai"
model = "gpt-4o-mini"

[[candidates]]
id = "gemma3-router"
provider = "ollama"
model = "gemma3:4b"
target_nodes = ["intent_router"]
```

Start with `target_nodes` on the simplest step. Swapping one step answers a more useful
question than swapping the whole agent, and failures are easier to read.

`temperature = 0` doesn't make a model deterministic on any backend, which is why each
case runs several times. Set `runs_per_case` in `[benchmark]`.

## Next

- [Reading results](reading-results.md)
- [Configuration](configuration.md)
- [CI](ci.md)
