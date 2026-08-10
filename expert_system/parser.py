"""Tokenizer and recursive-descent parser for the expert system's expression
language.

Precedence, tightest to loosest (per subject appendix VI.1):
    ()  >  !  >  +  >  |  >  ^  >  =>  >  <=>
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from expert_system.ast_nodes import And, ASTNode, Fact, Not, Or, Rule, Xor
from expert_system.errors import SyntaxError_


class TokKind(Enum):
    FACT = auto()
    NOT = auto()
    AND = auto()
    OR = auto()
    XOR = auto()
    LPAREN = auto()
    RPAREN = auto()
    IMPLIES = auto()
    IFF = auto()
    EOF = auto()


@dataclass
class Token:
    kind: TokKind
    text: str
    col: int


def tokenize(line: str, line_no: int) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if c.isspace():
            i += 1
            continue
        if c == "(":
            tokens.append(Token(TokKind.LPAREN, c, i))
            i += 1
        elif c == ")":
            tokens.append(Token(TokKind.RPAREN, c, i))
            i += 1
        elif c == "!":
            tokens.append(Token(TokKind.NOT, c, i))
            i += 1
        elif c == "+":
            tokens.append(Token(TokKind.AND, c, i))
            i += 1
        elif c == "|":
            tokens.append(Token(TokKind.OR, c, i))
            i += 1
        elif c == "^":
            tokens.append(Token(TokKind.XOR, c, i))
            i += 1
        elif c == "=":
            if i + 1 < n and line[i + 1] == ">":
                tokens.append(Token(TokKind.IMPLIES, "=>", i))
                i += 2
            else:
                raise SyntaxError_(
                    f"unexpected character '=' (did you mean '=>'?)", line_no, line
                )
        elif c == "<":
            if line[i : i + 3] == "<=>":
                tokens.append(Token(TokKind.IFF, "<=>", i))
                i += 3
            else:
                raise SyntaxError_(f"unexpected character '{c}'", line_no, line)
        elif c.isalpha():
            if not c.isupper():
                raise SyntaxError_(
                    f"facts must be uppercase letters, got '{c}'", line_no, line
                )
            tokens.append(Token(TokKind.FACT, c, i))
            i += 1
        else:
            raise SyntaxError_(f"unexpected character '{c}'", line_no, line)
    tokens.append(Token(TokKind.EOF, "", n))
    return tokens


class ExprParser:
    """Recursive-descent parser over a token list for one expression."""

    def __init__(self, tokens: list[Token], line_no: int, line_text: str):
        self.tokens = tokens
        self.pos = 0
        self.line_no = line_no
        self.line_text = line_text

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def error(self, message: str) -> SyntaxError_:
        return SyntaxError_(message, self.line_no, self.line_text)

    def parse_expression(self) -> ASTNode:
        node = self.parse_xor()
        return node

    def parse_xor(self) -> ASTNode:
        left = self.parse_or()
        while self.peek().kind == TokKind.XOR:
            self.advance()
            right = self.parse_or()
            left = Xor(left, right)
        return left

    def parse_or(self) -> ASTNode:
        left = self.parse_and()
        while self.peek().kind == TokKind.OR:
            self.advance()
            right = self.parse_and()
            left = Or(left, right)
        return left

    def parse_and(self) -> ASTNode:
        left = self.parse_not()
        while self.peek().kind == TokKind.AND:
            self.advance()
            right = self.parse_not()
            left = And(left, right)
        return left

    def parse_not(self) -> ASTNode:
        if self.peek().kind == TokKind.NOT:
            self.advance()
            return Not(self.parse_not())
        return self.parse_atom()

    def parse_atom(self) -> ASTNode:
        tok = self.peek()
        if tok.kind == TokKind.FACT:
            self.advance()
            return Fact(tok.text)
        if tok.kind == TokKind.LPAREN:
            self.advance()
            node = self.parse_expression()
            if self.peek().kind != TokKind.RPAREN:
                raise self.error("unclosed parenthesis")
            self.advance()
            return node
        if tok.kind == TokKind.EOF:
            raise self.error("unexpected end of expression")
        raise self.error(f"unexpected token '{tok.text}'")


def parse_rule_line(line: str, line_no: int, raw_line: str) -> Rule:
    """Parse a full rule line: <expr> => <expr>  or  <expr> <=> <expr>."""
    tokens = tokenize(line, line_no)

    split_idx = None
    op_kind = None
    depth = 0
    for idx, tok in enumerate(tokens):
        if tok.kind == TokKind.LPAREN:
            depth += 1
        elif tok.kind == TokKind.RPAREN:
            depth -= 1
        elif tok.kind in (TokKind.IMPLIES, TokKind.IFF) and depth == 0:
            if split_idx is not None:
                raise SyntaxError_(
                    "multiple '=>' / '<=>' operators in a single rule are not supported",
                    line_no,
                    raw_line,
                )
            split_idx = idx
            op_kind = tok.kind

    if split_idx is None:
        raise SyntaxError_(
            "rule line must contain '=>' or '<=>'", line_no, raw_line
        )

    left_tokens = tokens[:split_idx] + [Token(TokKind.EOF, "", 0)]
    right_tokens = tokens[split_idx + 1 :]

    if len(left_tokens) <= 1:
        raise SyntaxError_("missing condition before '=>'/'<=>'", line_no, raw_line)
    if len(right_tokens) <= 1:
        raise SyntaxError_("missing conclusion after '=>'/'<=>'", line_no, raw_line)

    cond_parser = ExprParser(left_tokens, line_no, raw_line)
    condition = cond_parser.parse_expression()
    if cond_parser.peek().kind != TokKind.EOF:
        raise cond_parser.error(
            f"unexpected token '{cond_parser.peek().text}' in condition"
        )

    concl_parser = ExprParser(right_tokens, line_no, raw_line)
    conclusion = concl_parser.parse_expression()
    if concl_parser.peek().kind != TokKind.EOF:
        raise concl_parser.error(
            f"unexpected token '{concl_parser.peek().text}' in conclusion"
        )

    return Rule(
        condition=condition,
        conclusion=conclusion,
        line_no=line_no,
        source_text=line.strip(),
        is_biconditional=(op_kind == TokKind.IFF),
    )
