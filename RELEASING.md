# Releasing

Publishing is a tag push. Everything else here is what has to be true before
that tag exists.

## Once, before the first release

On PyPI, add a **trusted publisher** for this repository: workflow
`release.yml`, environment `pypi`. That is the whole credential setup —
[`.github/workflows/release.yml`](.github/workflows/release.yml) has no
`password:` and this repository holds no PyPI token, because a token in
`secrets` is a credential any workflow that can read secrets can exfiltrate,
and that is the wrong trade for a tool people are asked to install into their
CI.

Rehearse it before trusting it:

```bash
gh workflow run release.yml      # dry-run defaults to true: builds and checks, publishes nothing
```

## Every release

1. **Decide the number.** While the major version is 0, a minor bump is the
   right answer for anything that changes the CLI surface, the stored schema,
   or the gate's exit codes. A patch bump is for fixes that leave all three
   alone.

2. **Bump it in one place.** `__version__` in
   [`src/slmeval/__init__.py`](src/slmeval/__init__.py); the build reads it
   from there and the release workflow refuses a tag that disagrees with it.

3. **Write the changelog entry** in [`CHANGELOG.md`](CHANGELOG.md). Name the
   behaviour that changed, not the commits. Anything touching exit codes or the
   report `schema` field says so explicitly — those two are contracts.

4. **Run what CI runs, locally.**

   ```bash
   python -m ruff check src tests scripts
   python -m mypy src
   python -m pytest
   ```

5. **Check the teardown still reproduces.** It is the first thing a stranger
   reads and it makes numeric claims about files committed next to it.

   ```bash
   python scripts/teardown.py --check
   ```

   A stale artifact means a number moved. Regenerate, read the diff, and decide
   whether it is a fix or a regression before committing it — then say which in
   the changelog.

6. **Use the artifact, not the working tree.** `pytest` runs against `src/` and
   so proves nothing about what `pip install slmeval` ships: a package missing
   from the wheel, a console script that is not executable on Windows.

   ```bash
   python scripts/install_smoke.py
   ```

   This builds the wheel, installs it into a throwaway venv with nothing else
   in it, and runs `init`, `validate`, `doctor`, `record`, `run`, `report`,
   `show` and `gate` as a new user would — including that a floor nothing can
   clear exits exactly `1`.

7. **Check the long description renders.** A broken README is not fixable
   in place on PyPI; it needs a new version.

   ```bash
   python -m build
   python -m twine check --strict dist/*
   ```

8. **Tag and push.**

   ```bash
   git tag v0.1.0 && git push origin v0.1.0
   ```

   The workflow then re-runs the offline suite and the install smoke **on the
   tagged tree** on Linux and Windows, checks the tag against `__version__`,
   builds, `twine check --strict`s, and publishes over OIDC. A tag can point
   anywhere, which is why none of that is taken on trust from the commit's own
   CI run.

## After

```bash
python -m venv /tmp/check && /tmp/check/bin/pip install slmeval
/tmp/check/bin/slmeval --version && /tmp/check/bin/slmeval doctor
```

`doctor` exiting `3` on a machine with no Ollama is correct, not a failed
release.

A version cannot be replaced on PyPI once uploaded, and cannot be re-uploaded
after deletion. If something shipped broken, the fix is the next patch version;
yank the bad one so a resolver stops choosing it while leaving it installable
for anyone who pinned it.
