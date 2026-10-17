"""Tests for HTML report rendering."""

from mutest.report import render_report


def _ctx():
    return {
        "target": "demo",
        "version": "0.1.0",
        "score": 0.75,
        "total": 4, "killed": 3, "survived": 1,
        "timeout": 0, "error": 0, "elapsed_s": 12.3,
        "files": [{"name": "calc.py", "total": 4, "killed": 3,
                   "survived": 1, "score": 0.75}],
        "survived_mutants": [{
            "id": "calc.py:14:ROR:1", "operator": "ROR",
            "description": ">= -> >",
            "original_snippet": "    return age >= 18",
            "mutated_snippet": "    return age > 18",
        }],
        "generated_tests": [{
            "name": "test_mutest_boundary", "code": "def test_mutest_boundary():\n    assert True\n",
            "target": "calc.py:14:ROR:1", "model": "fake", "status": "killed ✓",
        }],
    }


def test_report_renders(tmp_path):
    html = render_report(_ctx())
    assert "75.0%" in html
    assert "calc.py:14:ROR:1" in html
    assert "&gt;=" in html or ">=" in html  # snippet present (escaped or not)
    assert "test_mutest_boundary" in html
    assert "mutest 0.1.0" in html


def test_report_no_survivors_message():
    ctx = _ctx()
    ctx["survived"] = 0
    ctx["survived_mutants"] = []
    ctx["killed"] = 4
    ctx["score"] = 1.0
    html = render_report(ctx)
    assert "No survived mutants" in html


def test_report_without_ai_section():
    ctx = _ctx()
    ctx["generated_tests"] = []
    html = render_report(ctx)
    assert "AI-generated tests" not in html
