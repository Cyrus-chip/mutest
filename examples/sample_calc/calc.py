"""Tiny calculator — the mutest demo and self-test fixture.

The test suite below is deliberately weak: it never checks the boundary
``is_adult(18)``, so the ROR mutant (``>=`` -> ``>``) survives. That is
exactly the gap mutest's AI loop is designed to close.
"""


def add(a, b):
    return a + b


def is_adult(age):
    return age >= 18


def clamp(value, lo, hi):
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def discount(price, rate):
    if rate < 0 or rate > 1:
        raise ValueError("rate must be in [0, 1]")
    return price * (1 - rate)
