"""Tests for the parallel pytest runner."""

from pathlib import Path

import pytest

from mutest.mutators import generate_mutants
from mutest.runner import (
    KILLED,
    SURVIVED,
    BaselineError,
    RunSummary,
    check_baseline,
    run_mutants,
    sanitize_environment,
)

SAMPLE = Path(__file__).resolve().parent.parent / "examples" / "sample_calc"


def test_baseline_green():
    out = check_baseline(SAMPLE, SAMPLE, timeout=120)
    assert "passed" in out


def test_baseline_red_raises(tmp_path):
    src = tmp_path / "src"
    tests = tmp_path / "tests"
    src.mkdir()
    tests.mkdir()
    (src / "m.py").write_text("def f():\n    return 1\n")
    (tests / "test_m.py").write_text("def test_f():\n    assert False\n")
    with pytest.raises(BaselineError):
        check_baseline(src, tests, timeout=60)


def _mutants_by_desc():
    src = (SAMPLE / "calc.py").read_text()
    ms = generate_mutants(src, "calc.py")
    return {(m.description, m.lineno): m for m in ms}


def test_known_killed_mutant():
    # add(): AOR + -> - must be caught by test_add
    by = _mutants_by_desc()
    m = by[("+ -> -", 10)]
    results, _ = run_mutants([m], SAMPLE, SAMPLE, workers=1, timeout=120)
    assert results[0].status == KILLED


def test_known_survived_mutant():
    # is_adult(): ROR >= -> > survives because boundary 18 is untested
    by = _mutants_by_desc()
    m = by[(">= -> >", 14)]
    results, _ = run_mutants([m], SAMPLE, SAMPLE, workers=1, timeout=120)
    assert results[0].status == SURVIVED


def test_summary_score_math():
    by = _mutants_by_desc()
    mutants = [by[("+ -> -", 10)], by[(">= -> >", 14)]]
    _results, summary = run_mutants(mutants, SAMPLE, SAMPLE, workers=2, timeout=120)
    assert summary.total == 2
    assert summary.killed == 1
    assert summary.survived == 1
    assert summary.score == 0.5


def test_summary_score_counts_timeout_and_error_in_denominator():
    summary = RunSummary(total=4, killed=1, survived=1, timeout=1, error=1)
    assert summary.score == 0.25
    assert summary.as_dict()["score"] == 0.25


def test_secret_like_environment_is_removed_for_test_subprocesses():
    env = sanitize_environment({
        "OPENAI_API_KEY": "secret",
        "GITHUB_TOKEN": "token",
        "DATABASE_PASSWORD": "password",
        "SERVICE_KEY": "key",
        "SAFE_SETTING": "kept",
    })
    assert env == {"SAFE_SETTING": "kept"}


def test_worker_passes_sanitized_environment_to_pytest(monkeypatch):
    from mutest.runner import _Worker

    captured = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs["env"])
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("SAFE_SETTING", "kept")
    monkeypatch.setattr("mutest.runner.subprocess.run", fake_run)
    worker = _Worker(SAMPLE, SAMPLE)
    try:
        rc, _ = worker.run_pytest(timeout=1, sanitize_env=True)
    finally:
        worker.close()
    assert rc == 0
    assert "OPENAI_API_KEY" not in captured
    assert captured["SAFE_SETTING"] == "kept"


def test_parallel_results_deterministic():
    """Regression test: concurrent workers must not clobber each other's
    workspaces (task index != thread identity)."""
    from mutest.mutators import generate_mutants

    src = (SAMPLE / "calc.py").read_text()
    mutants = generate_mutants(src, "calc.py")
    seen = set()
    for _ in range(2):
        results, summary = run_mutants(mutants, SAMPLE, SAMPLE,
                                       workers=4, timeout=120)
        survived = tuple(sorted(r.mutant.id for r in results
                                if r.status == "survived"))
        seen.add((summary.score, survived))
    assert len(seen) == 1
    # ground truth established with workers=1
    assert seen.pop()[0] == 0.7


def test_worker_never_writes_pycache():
    """Regression test: stale .pyc files (CPython records source mtime with
    only second granularity) once caused one mutant's bytecode to shadow
    another's, producing flaky mutation scores. Workers must neither copy
    nor create __pycache__."""
    from mutest.mutators import generate_mutants
    from mutest.runner import _Worker

    src = (SAMPLE / "calc.py").read_text()
    mutants = generate_mutants(src, "calc.py")
    w = _Worker(SAMPLE, SAMPLE)
    try:
        assert list(w.root.rglob("__pycache__")) == []
        r = w.run_mutant(mutants[0], timeout=120)
        assert r.status in ("killed", "survived")
        assert list(w.root.rglob("__pycache__")) == [], \
            "worker must not create __pycache__ (PYTHONDONTWRITEBYTECODE)"
    finally:
        w.close()
