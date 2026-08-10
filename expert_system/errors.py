"""Custom exceptions for the expert system."""


class ExpertSystemError(Exception):
    """Base class for all errors this program reports to the user."""


class SyntaxError_(ExpertSystemError):
    """Raised on malformed input (bad tokens, unbalanced parens, ...)."""

    def __init__(self, message: str, line_no: int | None = None, line_text: str | None = None):
        self.line_no = line_no
        self.line_text = line_text
        location = f" (line {line_no})" if line_no is not None else ""
        text = f": {line_text!r}" if line_text else ""
        super().__init__(f"Syntax error{location}{text}: {message}")


class ContradictionError(ExpertSystemError):
    """Raised when the ruleset is provably self-contradictory for a fact."""

    def __init__(self, fact: str, detail: str = ""):
        self.fact = fact
        msg = f"Contradiction detected for fact '{fact}'"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)
