# Security policy

## Supported versions

slmeval is in alpha. Security fixes go into the latest release only.

## Reporting a vulnerability

Please don't open a public issue. Report it privately through
[GitHub's security advisories](https://github.com/mohitraj3697/slmeval/security/advisories/new).

Include the version, what you found, how to reproduce it and what an attacker could do with
it. You should hear back within a week.

## What to keep in mind when using slmeval

- **`slmeval record` runs your real tools.** Point them at a test environment if they have
  side effects.
- **Cassettes and traces contain real data.** `.slmeval/` holds your tools' responses and
  your prompts. Check them before committing them or attaching them to an issue.
- **`slmeval run` imports and executes the agent named in `eval.toml`.** Treat an `eval.toml`
  from someone else like any other code you would run, including in CI on pull requests from
  forks.
- **API keys belong in environment variables.** `eval.toml` stores only the variable's name.
