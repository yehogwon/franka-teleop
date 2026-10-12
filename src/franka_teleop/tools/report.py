"""PASS/WARN/FAIL reporting shared by the ft-check-* tools.

Text mode prints each result as soon as it is known (some checks take seconds)::

    PASS  kernel: PREEMPT_RT (6.8.0-rt8)
    FAIL  rtprio: RLIMIT_RTPRIO is 0, libfranka needs 99
          fix: ...

``--json`` mode prints nothing until the end, then one JSON object on stdout.
The exit code is 0 when no check failed, 1 otherwise.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TextIO

Status = Literal["PASS", "WARN", "FAIL"]
STATUSES: tuple[Status, ...] = ("PASS", "WARN", "FAIL")
HINT_INDENT = " " * 6


@dataclass
class Result:
    name: str
    status: Status
    detail: str
    hint: str = ""
    """How to fix it. Printed on WARN/FAIL only."""
    data: dict[str, Any] = field(default_factory=dict)
    """Machine-readable values, only in --json output."""

    def lines(self) -> list[str]:
        out = [f"{self.status:<4}  {self.name}: {self.detail}"]
        if self.hint and self.status != "PASS":
            hint = self.hint.splitlines()
            out.append(f"{HINT_INDENT}fix: {hint[0]}")
            out += [f"{HINT_INDENT}     {line}".rstrip() for line in hint[1:]]
        return out


def passed(name: str, detail: str, **data: Any) -> Result:
    return Result(name, "PASS", detail, data=data)


def warn(name: str, detail: str, hint: str = "", **data: Any) -> Result:
    return Result(name, "WARN", detail, hint, data=data)


def fail(name: str, detail: str, hint: str = "", **data: Any) -> Result:
    return Result(name, "FAIL", detail, hint, data=data)


class Report:
    def __init__(self, tool: str, as_json: bool = False, out: TextIO | None = None):
        self.tool = tool
        self.as_json = as_json
        self.out = out if out is not None else sys.stdout
        self.results: list[Result] = []
        self.notes: list[str] = []

    def add(self, result: Result) -> Result:
        self.results.append(result)
        if not self.as_json:
            for line in result.lines():
                print(line, file=self.out, flush=True)
        return result

    def note(self, text: str) -> None:
        """Informational line that is not a check (JSON: the "notes" list)."""
        self.notes.append(text)
        if not self.as_json:
            print(f"INFO  {text}", file=self.out, flush=True)

    @property
    def ok(self) -> bool:
        return all(r.status != "FAIL" for r in self.results)

    def counts(self) -> dict[str, int]:
        return {s: sum(r.status == s for r in self.results) for s in STATUSES}

    def get(self, name: str) -> Result | None:
        return next((r for r in self.results if r.name == name), None)

    def finish(self) -> int:
        """Print the summary (or the JSON document) and return the process exit code."""
        counts = self.counts()
        if self.as_json:
            doc = {
                "tool": self.tool,
                "ok": self.ok,
                "counts": counts,
                "results": [asdict(r) for r in self.results],
                "notes": self.notes,
            }
            print(json.dumps(doc, indent=2, default=_jsonable), file=self.out, flush=True)
        else:
            verdict = "NOT READY" if not self.ok else "OK, see warnings" if counts["WARN"] else "OK"
            summary = ", ".join(f"{counts[s]} {s}" for s in STATUSES)
            print(f"\n{self.tool}: {verdict} ({summary})", file=self.out, flush=True)
        return 0 if self.ok else 1


def _jsonable(obj: Any) -> Any:
    if hasattr(obj, "tolist"):  # numpy arrays and scalars
        return obj.tolist()
    return str(obj)
