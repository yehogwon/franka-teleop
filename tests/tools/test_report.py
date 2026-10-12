import io
import json

from franka_teleop.tools.report import Report, fail, passed, warn


def test_text_lines_and_exit_code():
    out = io.StringIO()
    report = Report("ft-x", out=out)
    report.add(passed("a", "fine", hint_never_shown=1))
    report.add(warn("b", "meh", "do this\nthen that"))
    report.note("just so you know")
    assert report.ok and report.finish() == 0
    lines = out.getvalue().splitlines()
    assert lines[0] == "PASS  a: fine"
    assert lines[1] == "WARN  b: meh"
    assert lines[2] == "      fix: do this"
    assert lines[3] == "           then that"
    assert lines[4] == "INFO  just so you know"
    assert lines[-1] == "ft-x: OK, see warnings (1 PASS, 1 WARN, 0 FAIL)"


def test_json_has_everything_and_nothing_else():
    out = io.StringIO()
    report = Report("ft-x", as_json=True, out=out)
    report.add(fail("c", "broken", "fix it", value=1.5))
    report.note("n")
    assert not report.ok and report.finish() == 1
    doc = json.loads(out.getvalue())
    assert doc["ok"] is False and doc["counts"] == {"PASS": 0, "WARN": 0, "FAIL": 1}
    result = {"name": "c", "status": "FAIL", "detail": "broken", "hint": "fix it"}
    assert doc["results"] == [{**result, "data": {"value": 1.5}}]
    assert doc["notes"] == ["n"]
