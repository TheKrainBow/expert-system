"""AST node definitions for propositional logic expressions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TriState(Enum):
    """Kleene three-valued logic."""

    TRUE = "True"
    FALSE = "False"
    UNDETERMINED = "Undetermined"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class ASTNode:
    """Base class for expression AST nodes."""


@dataclass(frozen=True)
class Fact(ASTNode):
    name: str

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return self.name


@dataclass(frozen=True)
class Not(ASTNode):
    operand: ASTNode

    def __repr__(self) -> str:  # pragma: no cover
        return f"!{self.operand!r}"


@dataclass(frozen=True)
class And(ASTNode):
    left: ASTNode
    right: ASTNode

    def __repr__(self) -> str:  # pragma: no cover
        return f"({self.left!r} + {self.right!r})"


@dataclass(frozen=True)
class Or(ASTNode):
    left: ASTNode
    right: ASTNode

    def __repr__(self) -> str:  # pragma: no cover
        return f"({self.left!r} | {self.right!r})"


@dataclass(frozen=True)
class Xor(ASTNode):
    left: ASTNode
    right: ASTNode

    def __repr__(self) -> str:  # pragma: no cover
        return f"({self.left!r} ^ {self.right!r})"


def format_expr(node: ASTNode) -> str:
    """Pretty-print `node` standalone (as it would read on its own line),
    dropping the one redundant outer paren repr() always adds for And/Or/
    Xor. Used wherever a condition/conclusion needs to be shown as its own
    clean rule text -- e.g. splitting a biconditional into two `=>` rules.
    """
    text = repr(node)
    if isinstance(node, (And, Or, Xor)) and text.startswith("(") and text.endswith(")"):
        return text[1:-1]
    return text


@dataclass
class Rule:
    """A single implication: condition => conclusion.

    Biconditional (<=>) rules are expanded by the file parser into two
    Rule objects (A=>B and B=>A) before reaching the engine, so the engine
    only ever deals with plain implications.
    """

    condition: ASTNode
    conclusion: ASTNode
    line_no: int
    source_text: str
    is_biconditional: bool = False

    def __repr__(self) -> str:  # pragma: no cover
        op = "<=>" if self.is_biconditional else "=>"
        return f"{self.condition!r} {op} {self.conclusion!r}"
