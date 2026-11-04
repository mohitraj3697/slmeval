# Releasing

A release is published by pushing a version tag.

## One-time setup

On PyPI, add a trusted publisher for this repository with workflow `release.yml` and
environment `pypi`. The repository holds no PyPI token.

Rehearse from **Actions → release → Run workflow**. Dry-run is on by default, so it builds and
checks but publishes nothing.

## Each release

1. Choose the version. While the major version is 0, use a minor bump for changes to the CLI,
   stored schema or exit codes, and a patch bump for anything else.
2. Set `__version__` in [`src/slmeval/__init__.py`](src/slmeval/__init__.py). The release workflow
   refuses a tag that disagrees with it.
3. Add a [`CHANGELOG.md`](CHANGELOG.md) entry. Call out any change to exit codes or the report
   `schema`.
4. Run the checks:

   ```bash
   python -m ruff check src tests scripts
   python -m mypy src
   python -m pytest
   ```

5. Check that the published teardown still reproduces. If it doesn't, a number moved: regenerate
   it, read the diff and say why in the changelog.

   ```bash
   python scripts/teardown.py --check
   python scripts/readme_visuals.py --check
   ```

6. Test the built wheel in a fresh environment:

   ```bash
   python scripts/install_smoke.py
   ```

7. Check that the package metadata renders. A broken PyPI page can only be fixed with a new version.

   ```bash
   python -m build
   python -m twine check --strict dist/*
   ```

8. Tag and push:

   ```bash
   git tag v0.1.0 && git push origin v0.1.0
   ```

   The workflow re-runs the test suite and the install check on the tagged commit, on Linux and
   Windows, then builds and publishes.

## Afterwards

Check the release from a clean environment. `doctor` exits 3 on a machine with no model server,
which is expected.

```bash
pip install slmeval
```

```bash
slmeval --version
```

```bash
slmeval doctor
```

A version number can't be reused on PyPI. If a release is broken, publish a patch version and yank
the bad one, so that resolvers skip it while anyone who pinned it can still install it.
