"""Tests for the AST mutation operators."""

import ast

from mutest.mutators import (
    ALL_OPERATORS,
    OPERATORS_BY_CODE,
    generate_mutants,
)


def _mutants_for(src: str, code: str):
    op = OPERATORS_BY_CODE[code]
    return generate_mutants(src, "x.py", [op])


def test_all_operators_registered():
    codes = [op.code for op in ALL_OPERATORS]
    assert codes == ["AOR", "ROR", "LOR", "NOT", "NEGCOND",
                     "NUM", "STR", "RETNONE", "RETBOOL", "UNARY"]
    assert len(set(codes)) == 10


def test_aor():
    ms = _mutants_for("def f(a, b):\n    return a + b\n", "AOR")
    assert len(ms) == 1
    assert ms[0].description == "+ -> -"
    assert "return a - b" in ms[0].source


def test_ror():
    ms = _mutants_for("def f(a):\n    return a >= 18\n", "ROR")
    assert len(ms) == 1
    assert ms[0].description == ">= -> >"
    assert "return a > 18" in ms[0].source


def test_ror_skips_chained_comparison():
    ms = _mutants_for("def f(a):\n    return 1 < a < 10\n", "ROR")
    assert ms == []


def test_lor():
    ms = _mutants_for("def f(a, b):\n    return a and b\n", "LOR")
    assert len(ms) == 1
    assert "return a or b" in ms[0].source


def test_not_removal():
    ms = _mutants_for("def f(a):\n    return not a\n", "NOT")
    assert len(ms) == 1
    assert "return a" in ms[0].source


def test_negcond():
    src = "def f(a):\n    if a > 1:\n        return 1\n    return 0\n"
    ms = _mutants_for(src, "NEGCOND")
    assert len(ms) == 1
    assert "if not a > 1:" in ms[0].source


def test_negcond_skips_already_negated():
    ms = _mutants_for("def f(a):\n    if not a:\n        return 1\n", "NEGCOND")
    assert ms == []


def test_num():
    ms = _mutants_for("def f():\n    return 42\n", "NUM")
    assert len(ms) == 1
    assert "return 0" in ms[0].source
    ms0 = _mutants_for("def f():\n    return 0\n", "NUM")
    assert "return 1" in ms0[0].source


def test_num_skips_booleans():
    # True/False are ints in Python; they belong to RETBOOL, not NUM
    ms = _mutants_for("def f():\n    return True\n", "NUM")
    assert ms == []


def test_str():
    ms = _mutants_for('def f():\n    return "hello"\n', "STR")
    assert len(ms) == 1
    assert 'return ""' in ms[0].source or "return ''" in ms[0].source


def test_str_skips_docstrings():
    src = 'def f():\n    """Docstring."""\n    return "real"\n'
    ms = _mutants_for(src, "STR")
    assert len(ms) == 1
    assert "real" not in ms[0].source or '"""Docstring."""' in ms[0].source
    # the docstring itself must be untouched
    assert '"""Docstring."""' in ms[0].source or "'''Docstring.'''" in ms[0].source \
        or '"""Docstring."""' in ms[0].source


def test_retnone():
    ms = _mutants_for("def f(a):\n    return a + 1\n", "RETNONE")
    assert len(ms) == 1
    assert "return None" in ms[0].source


def test_retnone_skips_bare_and_none_return():
    ms = _mutants_for("def f():\n    return\n", "RETNONE")
    assert ms == []
    ms = _mutants_for("def f():\n    return None\n", "RETNONE")
    assert ms == []


def test_retbool():
    ms = _mutants_for("def f():\n    return True\n", "RETBOOL")
    assert len(ms) == 1
    assert "return False" in ms[0].source


def test_unary():
    ms = _mutants_for("def f(a):\n    return -a\n", "UNARY")
    assert len(ms) == 1
    assert "return a" in ms[0].source


def test_mutants_are_valid_python():
    src = (
        "def f(a, b):\n"
        "    if a > b and not b:\n"
        "        return a - b\n"
        "    return 'x'\n"
    )
    ms = generate_mutants(src, "x.py")
    assert len(ms) > 5
    for m in ms:
        ast.parse(m.source)  # must not raise
        assert m.source != src


def test_no_duplicate_mutants():
    src = "def f(a):\n    return a + a\n"
    ms = generate_mutants(src, "x.py")
    ids = [m.id for m in ms]
    assert len(ids) == len(set(ids))


def test_syntax_error_source_yields_nothing():
    assert generate_mutants("def broken(:\n", "x.py") == []
