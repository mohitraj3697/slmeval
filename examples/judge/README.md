# Calibrating the judge

A judge produces a number. The only thing that makes that number evidence is a
measured agreement with human labels — so `slmeval` will not run one until you
have measured it.

```bash
slmeval judge calibrate labels.jsonl
```

That grades every labelled pair with your configured judge model, compares the
verdicts with yours, and writes `slmeval.judge.json`. **Commit that file**, for
the same reason you commit `slmeval.baseline.json`: "we accepted a judge that
agrees with us 71% of the time" should be a diff somebody approved, not a
number in one laptop's cache.

## The labels are yours

[`labels.example.jsonl`](labels.example.jsonl) is **illustrative**. Every line
carries `"source": "example"`, and a calibration computed from it reports
itself as a smoke test of the harness rather than a measurement of a judge.
Shipping fabricated labels as if they were human ones would undermine the only
argument for having a judge at all — that somebody checked it.

Copy it, re-label every line yourself, and set `"source": "human"`:

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

Blank lines and `//` comments are skipped. Twenty labels is the floor — below
that the command says so and refuses to call the result a calibration — and
thirty is a reasonable target.

Label the *hard* ones. Twenty-four obvious passes measure nothing; the pairs
worth labelling are the terse-but-correct answer, the padded-but-correct
answer, the one that is right in the wrong language, and the one that did the
thing you forbade and then asked permission. The example set is built that way
on purpose.

## Reading the output

This is the real output of the scripted judge below against
`labels.example.jsonl`:

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

**Read the interval, not the rate.** Twenty-four labels put a 41.7% point
estimate anywhere between 24% and 61%, and the floor in
`[judge].min_agreement` is read against the lower bound — the same rule as
`pass^k`, for the same reason.

**Read kappa next to it.** This judge says "pass" to everything, and the whole
`pass` column proves it. Its 41.7% is exactly what chance predicts given how
these labels are split, so kappa is 0.00. On a friendlier label set — nine
pairs in ten passing — the same do-nothing judge would have scored 90%
agreement and still known nothing. That is the number kappa exists to deflate,
and it is why neither figure is ever printed without the other.

**Then read the disagreements**, which the command lists in full. That is the
actual deliverable: those pairs tell you whether the rubric is ambiguous, the
judge is lenient, or you labelled one wrong.

## Configuration

```toml
[judge]
provider = "openai"
model = "gpt-4o-mini"
api_key_env = "OPENAI_API_KEY"
min_agreement = 0.7      # on the Wilson lower bound
```

No `[judge]` section means no judge, and Tier 3 simply does not run. Tiers 1
and 2 are free, cost nothing to re-run, and carry most of the signal; a judge is
for the remainder.

### Offline, with no key

```toml
[judge]
provider = "fake"
model = "scripted-judge"
extra_body.script = [{ text = '{"verdict": "pass", "score": 0.9, "reasoning": "names the order, the carrier and a date", "cited_span": "ORD-9921"}' }]
```

A scripted judge that answers "pass" to everything, which is exactly the judge
kappa is designed to catch — run it against the example labels and watch the
agreement look respectable while kappa sits near zero. That is the lesson, and
it costs nothing to see.

## What the judge is not allowed to do

- **It never grades a run that already failed.** Tiers 1 and 2 go first; a
  judge call happens only when both passed and the case carries a
  `judge_rubric`. Paying to grade a known failure is pure cost.
- **It never learns which model wrote which answer.** Answers go in as a
  randomised A/B pair with neutral labels, in an order derived from the case
  seed so it is reproducible and identical across candidates.
- **It never gets paid twice for the same question.** Judgements are cached by
  `hash(rubric, reference, candidate, judge_model)` in `.slmeval/runs.sqlite`,
  so re-rendering a report or re-running a gate costs nothing.
- **It never answers in prose.** The verdict is a schema and goes through the
  same structured-output ladder as any other call, so a judge that cannot emit
  clean JSON is reported as such rather than regex-parsed.
- **`unclear` is a real answer.** A judge forced to choose on an answer it
  cannot assess will pick one, and that coin flip would enter the scorecard as
  a measurement. An abstention is reported as unjudged and kept out of the
  rate.

## Status

The judge is a skeleton: the schema, the blinding, the cache and this
calibration harness are built and tested, and Tier 3 is **not yet wired into
`slmeval run`**. That order is deliberate. The thing that decides whether a
judge is worth running is a calibration figure from real labels, so the harness
that produces it ships first.
