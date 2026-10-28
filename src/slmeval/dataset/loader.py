"""Read and validate a JSONL golden dataset.

Every problem in the file is collected before raising, rather than failing on
the first one. For a data file that is the right trade: the user fixes
everything in one pass instead of running the command eleven times.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from slmeval.config.loader import format_loc
from slmeval.dataset.models import Case
from slmeval.errors import DatasetError, Problem


@dataclass
class Dataset:
    path: Path
    cases: list[Case]
    content_hash: str

    def __len__(self) -> int:
        return len(self.cases)

    @property
    def tags(self) -> list[str]:
        return sorted({t for c in self.cases for t in c.tags})

    def case(self, case_id: str) -> Case | None:
        return next((c for c in self.cases if c.case_id == case_id), None)


def load_dataset(path: Path) -> Dataset:
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise DatasetError(f"could not read dataset {path}: {exc}") from exc

    problems: list[Problem] = []
    cases: list[Case] = []
    seen: dict[str, int] = {}

    for lineno, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue

        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            problems.append(
                Problem(path=f"line {lineno}", message=f"invalid JSON: {exc.msg}", line=lineno)
            )
            continue

        if not isinstance(payload, dict):
            problems.append(
                Problem(
                    path=f"line {lineno}",
                    message=f"expected a JSON object, got {type(payload).__name__}",
                    line=lineno,
                )
            )
            continue

        try:
            case = Case.model_validate(payload)
        except ValidationError as exc:
            for err in exc.errors():
                problems.append(
                    Problem(
                        path=f"line {lineno}: {format_loc(tuple(err['loc']))}",
                        message=err["msg"].removeprefix("Value error, "),
                        line=lineno,
                    )
                )
            continue

        if case.case_id in seen:
            problems.append(
                Problem(
                    path=f"line {lineno}: case_id",
                    message=f"duplicate case_id {case.case_id!r} (first seen on line "
                    f"{seen[case.case_id]}) - ids key every result row and must be unique",
                    line=lineno,
                )
            )
            continue

        seen[case.case_id] = lineno
        cases.append(case)

    if problems:
        raise DatasetError(
            f"{path.name} has {len(problems)} problem{'s' if len(problems) != 1 else ''}",
            source=path,
            problems=problems,
        )

    if not cases:
        raise DatasetError(
            f"{path.name} contains no cases",
            source=path,
            hint="each line must be one JSON object with case_id, input and assert",
        )

    return Dataset(path=path, cases=cases, content_hash=dataset_hash(cases))


def dataset_hash(cases: list[Case]) -> str:
    """Content hash of the *parsed* cases, ordered as written.

    Hashing the parsed form rather than the file bytes means reformatting the
    JSONL does not change the identity of the dataset, while changing any
    assertion does. Recorded with every run so `compare` can refuse to put two
    runs side by side when they were measured against different cases — which
    is the cheapest possible guard against publishing a misleading benchmark.
    """
    canonical = "\n".join(
        json.dumps(c.model_dump(mode="json", by_alias=True), sort_keys=True, separators=(",", ":"))
        for c in cases
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
