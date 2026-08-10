"""Parses a full expert-system input file into rules, initial facts and queries."""

from __future__ import annotations

from dataclasses import dataclass, field

from expert_system.ast_nodes import Rule, format_expr
from expert_system.errors import SyntaxError_
from expert_system.parser import parse_rule_line


@dataclass
class ParsedInput:
    rules: list[Rule] = field(default_factory=list)
    initial_facts: set[str] = field(default_factory=set)
    queries: list[str] = field(default_factory=list)
    facts_line_seen: bool = False
    queries_line_seen: bool = False


def strip_comment(line: str) -> str:
    idx = line.find("#")
    return line if idx == -1 else line[:idx]


def parse_input_text(text: str) -> ParsedInput:
    result = ParsedInput()
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = strip_comment(raw_line)
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("="):
            facts_part = stripped[1:].strip()
            for ch in facts_part:
                if ch.isspace():
                    continue
                if not (ch.isalpha() and ch.isupper()):
                    raise SyntaxError_(
                        f"initial facts must be uppercase letters, got '{ch}'",
                        line_no,
                        raw_line,
                    )
                result.initial_facts.add(ch)
            result.facts_line_seen = True
            continue

        if stripped.startswith("?"):
            queries_part = stripped[1:].strip()
            if not queries_part:
                raise SyntaxError_("empty query line", line_no, raw_line)
            for ch in queries_part:
                if ch.isspace():
                    continue
                if not (ch.isalpha() and ch.isupper()):
                    raise SyntaxError_(
                        f"queries must be uppercase letters, got '{ch}'",
                        line_no,
                        raw_line,
                    )
                result.queries.append(ch)
            result.queries_line_seen = True
            continue

        rule = parse_rule_line(stripped, line_no, raw_line)
        result.rules.append(rule)

    if not result.queries_line_seen:
        raise SyntaxError_("input file has no query line (starting with '?')", None, None)

    return result


def expand_biconditionals(rules: list[Rule]) -> list[Rule]:
    """Split each `A <=> B` rule into `A => B` and `B => A`.

    Both resulting rules get their own clean `=>`-only source_text -- never
    the original `<=>` text with a "(reverse of...)" suffix tacked on. That
    matters beyond cosmetics: this text is what shows up as the rule's label
    everywhere (the editor's rule list, the resolution-tree graph), and a
    label like "A <=> D (reverse of <=>)" reads as a still-biconditional
    rule when it's really just "D => A", which is confusing to a reader
    trying to follow the actual reasoning step by step.
    """
    expanded: list[Rule] = []
    for rule in rules:
        if rule.is_biconditional:
            cond_text = format_expr(rule.condition)
            concl_text = format_expr(rule.conclusion)
            expanded.append(
                Rule(
                    condition=rule.condition,
                    conclusion=rule.conclusion,
                    line_no=rule.line_no,
                    source_text=f"{cond_text} => {concl_text}",
                    is_biconditional=True,
                )
            )
            expanded.append(
                Rule(
                    condition=rule.conclusion,
                    conclusion=rule.condition,
                    line_no=rule.line_no,
                    source_text=f"{concl_text} => {cond_text}",
                    is_biconditional=True,
                )
            )
        else:
            expanded.append(rule)
    return expanded


def parse_input_file(path: str) -> ParsedInput:
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as exc:
        raise SyntaxError_(f"cannot read input file '{path}': {exc.strerror}") from exc
    return parse_input_text(text)
