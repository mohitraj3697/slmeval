# Calibrating the judge

Tier 3 is an LLM judge for what assertions can't express. slmeval won't run one until you have
measured how often it agrees with your own labels.

```bash
slmeval judge calibrate labels.jsonl
```

This grades every labelled pair with your judge model, compares the verdicts with yours and writes
`slmeval.judge.json`. Commit that file, so that accepting a judge's agreement rate goes through review.

## Labels

[`labels.example.jsonl`](labels.example.jsonl) is only an illustration. Every line has
`"source": "example"`, and a calibration computed from it reports itself as a test of the harness.

Copy it, label every line yourself and set `"source": "human"`:

```json
{"case_id": "status_01",
 "question": "Where is my order ORD-9921?",
 "rubric": "The answer must name the order id, the carrier, and a delivery estimate.",
 "reference": "Your order ORD-9921 is with DHL and is due to arrive on 2026-02-03.",
 "candidate": "ORD-9921 is on its way with DHL and should reach you on 3 February.",
 "verdict": "pass",
 "note": "all three facts, different phrasing",
 "source": "human"}
```

Blank lines and `//` comments are skipped. Twenty labels is the minimum, and below that the command
won't call the result a calibration. Thirty is a reasonable target.

Label the hard cases: the terse but correct answer, the padded but correct answer, the answer in the
wrong language, and the one that did something forbidden and then asked permission. Obvious passes
tell you very little.

## Reading the output

This is real output from the scripted judge below, run against `labels.example.jsonl`:

```text
  judge      fake:scripted-judge
  labels     24 (fail 12, pass 10, unclear 2)
  agreement  41.7%  [24%, 61%]
  kappa      0.00
  verdict    NOT a calibration - these labels are not human labels

   human (rows) vs judge
         (columns)
         pass  fail  unclear
  pass     10     0        0
  fail     12     0        0
  unclear   2     0        0
```

**Check the interval.** With 24 labels, 41.7% could be anywhere from 24% to 61%. The floor in
`[judge].min_agreement` is compared with the lower bound.

**Check kappa.** This judge says "pass" to everything. Its 41.7% is what chance predicts for this mix
of labels, so kappa is 0.00. On a set where nine in ten pairs pass, the same judge would score 90%
agreement and still have a kappa of zero.

**Read the disagreements**, which the command lists in full. They show whether the rubric is ambiguous,
the judge is lenient or one of your labels is wrong.

## Configuration

```toml
[judge]
provider = "openai"
model = "gpt-4o-mini"
api_key_env = "OPENAI_API_KEY"
min_agreement = 0.7      # compared with the Wilson lower bound
```

With no `[judge]` section, Tier 3 doesn't run.

### Offline, with no key

```toml
[judge]
provider = "fake"
model = "scripted-judge"
extra_body.script = [{ text = '{"verdict": "pass", "score": 0.9, "reasoning": "names the order, the carrier and a date", "cited_span": "ORD-9921"}' }]
```

This scripted judge says "pass" to everything, which is the behaviour kappa is meant to expose.

## How the judge behaves

- **It doesn't grade runs that already failed.** It runs only when tiers 1 and 2 passed and the case has
  a `judge_rubric`.
- **It doesn't see which model wrote which answer.** Answers arrive as a randomised A/B pair with neutral
  labels, in an order derived from the case seed.
- **It caches its judgements**, keyed on `hash(rubric, reference, candidate, judge_model)` in
  `.slmeval/runs.sqlite`, so re-rendering a report costs nothing.
- **It answers in a schema**, through the same structured-output ladder as any other call.
- **`unclear` is allowed.** An abstention is reported as unjudged and left out of the rate.

## Status

The schema, blinding, cache and calibration harness are built and tested. Tier 3 isn't connected to
`slmeval run` yet. That is waiting on a calibration figure from real labels.
