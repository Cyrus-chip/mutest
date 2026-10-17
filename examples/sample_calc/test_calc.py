"""Deliberately weak test suite for the mutest demo.

Boundary case ``is_adult(18)`` is NOT tested on purpose.
"""

from calc import add, clamp, discount, is_adult


def test_add():
    assert add(1, 2) == 3


def test_is_adult():
    assert is_adult(20) is True
    assert is_adult(10) is False


def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-3, 0, 10) == 0
    assert clamp(99, 0, 10) == 10


def test_discount():
    assert discount(100, 0.2) == 80
