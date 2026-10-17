"""Tests for the LLM module (offline — no real API calls)."""

from mutest.llm import (
    FakeProvider,
    build_generation_prompt,
    extract_python_code,
)
from mutest.mutators import generate_mutants


def _ror_mutant():
    ms = generate_mutants("def is_adult(age):\n    return age >= 18\n", "m.py")
    return [m for m in ms if m.operator == "ROR"][0]


def test_extract_python_code_fenced():
    text = "Here you go:\n```python\ndef test_x():\n    assert True\n```\nDone."
    assert extract_python_code(text) == "def test_x():\n    assert True\n"


def test_extract_python_code_no_fence():
    assert extract_python_code("no code here") == ""


def test_generation_prompt_contains_mutant_info():
    m = _ror_mutant()
    msgs = build_generation_prompt(m, "src", "tests")
    assert len(msgs) == 2
    blob = msgs[1]["content"]
    assert "ROR" in blob
    assert ">= -> >" in blob


def test_fake_provider_canned():
    m = _ror_mutant()
    p = FakeProvider({m.id: "def test_x():\n    assert True\n"})
    gen = p.generate_test(m, "", "", "fake")
    assert gen.code == "def test_x():\n    assert True\n"
    assert gen.target_mutant_id == m.id


def test_fake_provider_placeholder_for_unknown():
    m = _ror_mutant()
    p = FakeProvider({})
    gen = p.generate_test(m, "", "", "fake")
    assert "placeholder" in gen.code


def test_fake_judge_always_approves():
    m = _ror_mutant()
    p = FakeProvider({})
    ok, _reason = p.judge(m, "def test_x():\n    assert True\n", "fake")
    assert ok is True
