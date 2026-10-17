"""AST-based mutation operators.

Each operator finds candidate sites in a parsed module and applies exactly
one mutation to a deep copy of the tree. The original tree is never touched,
and every generated mutant is re-parsed to guard against producing invalid
code.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field


@dataclass
class Mutant:
    """A single mutated variant of one source file."""

    id: str
    file: str  # path relative to the source dir
    lineno: int
    operator: str  # e.g. "AOR"
    title: str  # human readable operator name
    description: str  # e.g. "+ -> -"
    original_snippet: str
    mutated_snippet: str
    source: str  # full mutated module source


def _replace_node(tree: ast.AST, target: ast.AST, replacement: ast.AST) -> bool:
    """Replace the first occurrence of *target* (by identity) in *tree*."""
    for node in ast.walk(tree):
        for fname, value in ast.iter_fields(node):
            if value is target:
                setattr(node, fname, replacement)
                return True
            if isinstance(value, list):
                for i, item in enumerate(value):
                    if item is target:
                        value[i] = replacement
                        return True
    return False


def _locate(tree: ast.AST, target: ast.AST) -> ast.AST | None:
    """Find the node in *tree* at the same type+position as *target*.

    Positions are unique within one tree, so (type, lineno, col_offset) is
    a reliable key for mapping a node from the original tree to its deep
    copy.
    """
    key = (
        type(target).__name__,
        getattr(target, "lineno", -1),
        getattr(target, "col_offset", -1),
    )
    for node in ast.walk(tree):
        if (
            type(node).__name__,
            getattr(node, "lineno", -1),
            getattr(node, "col_offset", -1),
        ) == key:
            return node
    return None


def _parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    pmap: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            pmap[id(child)] = node
    return pmap


def _is_docstring(node: ast.Constant, parents: dict[int, ast.AST]) -> bool:
    parent = parents.get(id(node))
    if not isinstance(parent, ast.Expr) or parent.value is not node:
        return False
    grand = parents.get(id(parent))
    body = getattr(grand, "body", None)
    return isinstance(body, list) and bool(body) and body[0] is parent


def _op_symbol(op: ast.AST) -> str:
    return {
        ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
        ast.Mod: "%", ast.FloorDiv: "//", ast.Pow: "**",
        ast.Gt: ">", ast.Lt: "<", ast.GtE: ">=", ast.LtE: "<=",
        ast.Eq: "==", ast.NotEq: "!=",
        ast.And: "and", ast.Or: "or",
        ast.Not: "not", ast.USub: "-", ast.UAdd: "+",
    }.get(type(op), type(op).__name__)


class Operator:
    """Base class for a single mutation operator."""

    code: str = "XXX"
    title: str = "base operator"

    def find(self, tree: ast.AST, parents: dict[int, ast.AST]) -> list[ast.AST]:
        raise NotImplementedError

    def describe(self, node: ast.AST) -> str:
        raise NotImplementedError

    def mutate(self, tree: ast.AST, node: ast.AST) -> ast.AST:
        """Apply the mutation to *node* (located inside *tree*).

        May mutate in place or return a new tree.
        """
        raise NotImplementedError


class AOR(Operator):
    """Arithmetic operator replacement: + -> - etc."""

    code = "AOR"
    title = "Arithmetic operator replacement"
    _map = {
        ast.Add: ast.Sub, ast.Sub: ast.Add,
        ast.Mult: ast.Div, ast.Div: ast.Mult,
        ast.Mod: ast.FloorDiv, ast.FloorDiv: ast.Mod,
        ast.Pow: ast.Mult,
    }

    def find(self, tree, parents):
        return [
            n for n in ast.walk(tree)
            if isinstance(n, ast.BinOp) and type(n.op) in self._map
        ]

    def describe(self, node):
        return f"{_op_symbol(node.op)} -> {_op_symbol(self._map[type(node.op)]())}"

    def mutate(self, tree, node):
        node.op = self._map[type(node.op)]()
        return tree


class ROR(Operator):
    """Relational operator replacement: > -> >= etc."""

    code = "ROR"
    title = "Relational operator replacement"
    _map = {
        ast.Gt: ast.GtE, ast.Lt: ast.LtE,
        ast.GtE: ast.Gt, ast.LtE: ast.Lt,
        ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
    }

    def find(self, tree, parents):
        return [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Compare) and len(n.ops) == 1
            and type(n.ops[0]) in self._map
        ]

    def describe(self, node):
        return f"{_op_symbol(node.ops[0])} -> {_op_symbol(self._map[type(node.ops[0])]())}"

    def mutate(self, tree, node):
        node.ops[0] = self._map[type(node.ops[0])]()
        return tree


class LOR(Operator):
    """Logical operator replacement: and <-> or."""

    code = "LOR"
    title = "Logical operator replacement"
    _map = {ast.And: ast.Or, ast.Or: ast.And}

    def find(self, tree, parents):
        return [
            n for n in ast.walk(tree)
            if isinstance(n, ast.BoolOp) and type(n.op) in self._map
        ]

    def describe(self, node):
        return f"{_op_symbol(node.op)} -> {_op_symbol(self._map[type(node.op)]())}"

    def mutate(self, tree, node):
        node.op = self._map[type(node.op)]()
        return tree


class NOT(Operator):
    """Remove a `not`: `not x` -> `x`."""

    code = "NOT"
    title = "Remove logical negation"

    def find(self, tree, parents):
        return [
            n for n in ast.walk(tree)
            if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not)
        ]

    def describe(self, node):
        return "not x -> x"

    def mutate(self, tree, node):
        replacement = copy.deepcopy(node.operand)
        if not _replace_node(tree, node, replacement):
            raise RuntimeError("NOT: failed to locate target node")
        return tree


class NEGCOND(Operator):
    """Negate a branch condition: `if c:` -> `if not c:`."""

    code = "NEGCOND"
    title = "Negate branch condition"

    def find(self, tree, parents):
        return [
            n for n in ast.walk(tree)
            if isinstance(n, (ast.If, ast.While))
            and not (isinstance(n.test, ast.UnaryOp) and isinstance(n.test.op, ast.Not))
        ]

    def describe(self, node):
        return "if c: -> if not c:"

    def mutate(self, tree, node):
        node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
        return tree


class NUM(Operator):
    """Numeric constant replacement: 0 -> 1, n -> 0."""

    code = "NUM"
    title = "Numeric constant replacement"

    def find(self, tree, parents):
        out = []
        for n in ast.walk(tree):
            if not isinstance(n, ast.Constant):
                continue
            if isinstance(n.value, bool):
                continue  # booleans are handled by RETBOOL
            if isinstance(n.value, (int, float)) and not isinstance(n.value, complex):
                out.append(n)
        return out

    def describe(self, node):
        return f"{node.value!r} -> {self._replacement(node.value)!r}"

    @staticmethod
    def _replacement(v):
        return 1 if v == 0 else 0

    def mutate(self, tree, node):
        node.value = self._replacement(node.value)
        return tree


class STR(Operator):
    """String constant replacement: non-empty string -> empty string."""

    code = "STR"
    title = "String constant replacement"

    def find(self, tree, parents):
        return [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and n.value != "" and not _is_docstring(n, parents)
        ]

    def describe(self, node):
        return f"{node.value!r} -> ''"

    def mutate(self, tree, node):
        node.value = ""
        return tree


class RETNONE(Operator):
    """Return value replacement: `return x` -> `return None`."""

    code = "RETNONE"
    title = "Return None instead of value"

    def find(self, tree, parents):
        out = []
        for n in ast.walk(tree):
            if not isinstance(n, ast.Return) or n.value is None:
                continue
            if isinstance(n.value, ast.Constant) and n.value.value is None:
                continue
            if isinstance(n.value, ast.Constant) and isinstance(n.value.value, bool):
                continue  # RETBOOL handles booleans
            out.append(n)
        return out

    def describe(self, node):
        return "return x -> return None"

    def mutate(self, tree, node):
        node.value = ast.Constant(value=None)
        return tree


class RETBOOL(Operator):
    """Boolean return flip: `return True` -> `return False`."""

    code = "RETBOOL"
    title = "Flip boolean return value"

    def find(self, tree, parents):
        return [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Return)
            and isinstance(n.value, ast.Constant)
            and isinstance(n.value.value, bool)
        ]

    def describe(self, node):
        return f"return {node.value.value} -> return {not node.value.value}"

    def mutate(self, tree, node):
        node.value = ast.Constant(value=not node.value.value)
        return tree


class UNARY(Operator):
    """Remove unary minus/plus: `-x` -> `x`."""

    code = "UNARY"
    title = "Remove unary operator"

    def find(self, tree, parents):
        return [
            n for n in ast.walk(tree)
            if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd))
        ]

    def describe(self, node):
        return f"{_op_symbol(node.op)}x -> x"

    def mutate(self, tree, node):
        replacement = copy.deepcopy(node.operand)
        if not _replace_node(tree, node, replacement):
            raise RuntimeError("UNARY: failed to locate target node")
        return tree


ALL_OPERATORS: list[Operator] = [
    AOR(), ROR(), LOR(), NOT(), NEGCOND(),
    NUM(), STR(), RETNONE(), RETBOOL(), UNARY(),
]

OPERATORS_BY_CODE: dict[str, Operator] = {op.code: op for op in ALL_OPERATORS}


def _snippet(lines: list[str], lineno: int, context: int = 2) -> str:
    start = max(0, lineno - 1 - context)
    end = min(len(lines), lineno + context)
    return "\n".join(lines[start:end])


def generate_mutants(
    source: str,
    filename: str,
    operators: list[Operator] | None = None,
) -> list[Mutant]:
    """Generate all mutants for one module's source code."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    parents = _parent_map(tree)
    ops = operators or ALL_OPERATORS
    src_lines = source.splitlines()
    mutants: list[Mutant] = []
    seen: set[tuple] = set()
    counters: dict[tuple, int] = {}
    for op in ops:
        try:
            sites = op.find(tree, parents)
        except Exception:
            continue
        for node in sites:
            lineno = getattr(node, "lineno", 1) or 1
            t2 = copy.deepcopy(tree)
            target = _locate(t2, node)
            if target is None:
                continue
            try:
                new_tree = op.mutate(t2, target) or t2
                ast.fix_missing_locations(new_tree)
                new_source = ast.unparse(new_tree)
                ast.parse(new_source)  # sanity: must be valid Python
            except Exception:
                continue
            if new_source == source:
                continue  # mutation was a no-op
            key = (lineno, op.code, _snippet(new_source.splitlines(), lineno))
            if key in seen:
                continue
            seen.add(key)
            counters[(lineno, op.code)] = counters.get((lineno, op.code), 0) + 1
            n = counters[(lineno, op.code)]
            new_lines = new_source.splitlines()
            mutants.append(
                Mutant(
                    id=f"{filename}:{lineno}:{op.code}:{n}",
                    file=filename,
                    lineno=lineno,
                    operator=op.code,
                    title=op.title,
                    description=op.describe(node),
                    original_snippet=_snippet(src_lines, lineno),
                    mutated_snippet=_snippet(new_lines, lineno),
                    source=new_source,
                )
            )
    mutants.sort(key=lambda m: (m.file, m.lineno, m.operator))
    return mutants
