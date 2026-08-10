#!/usr/bin/env python3
"""Expert system CLI: backward-chaining inference engine for propositional logic.

Usage:
    python expert.py <input_file> [--interactive] [--why] [--graph]
"""

from __future__ import annotations

import argparse
import sys

from expert_system.engine import Engine
from expert_system.errors import ExpertSystemError
from expert_system.input_file import ParsedInput, expand_biconditionals, parse_input_file


def report_consistency(engine: Engine) -> str | None:
    """Run the ruleset-wide contradiction check and report it if it fails.

    Some contradictions only show up once every fact has settled to its
    final value (see Engine.check_consistency's docstring for why a single
    query can miss one) -- this runs that check up front so it's reported
    even if the fact it involves isn't among the ones queried. Returns the
    printed error message (so run_queries can avoid repeating it for a
    fact that's also directly queried), or None if the ruleset is clean.
    """
    try:
        engine.check_consistency()
        return None
    except ExpertSystemError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return str(exc)


def run_queries(engine: Engine, queries: list[str], show_why: bool, quiet_error: str | None = None) -> int:
    """Run all queries, printing results. Returns process exit status.

    `quiet_error` suppresses re-printing a contradiction that
    `check_consistency` already reported for the whole ruleset -- other,
    unaffected queries still print normally.
    """
    status = 0
    for fact in queries:
        if show_why:
            engine.trace.clear()  # only show the recursion tree for this query
        try:
            result = engine.evaluate(fact)
        except ExpertSystemError as exc:
            if str(exc) != quiet_error:
                print(f"error: {exc}", file=sys.stderr)
            status = 1
            continue
        print(f"{fact} = {result.value}")
        if show_why:
            for event in engine.trace:
                print("  " + ("  " * event.depth) + event.text)
    return status


def run_interactive(parsed: ParsedInput, show_why: bool) -> int:
    print("Interactive mode. Commands:")
    print("  =<FACTS>   set the initial facts (e.g. =ABG, or = for none)")
    print("  ?<FACTS>   query facts (e.g. ?GVX)")
    print("  quit       exit")
    facts = set(parsed.initial_facts)
    rules = expand_biconditionals(parsed.rules)
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            print()
            break
        if not line or line == "quit":
            break
        if line.startswith("="):
            facts = {c for c in line[1:].strip() if c.isalpha() and c.isupper()}
            print(f"initial facts set to: {''.join(sorted(facts)) or '(none)'}")
            continue
        if line.startswith("?"):
            queries = [c for c in line[1:].strip() if c.isalpha() and c.isupper()]
            engine = Engine(rules, facts)
            quiet_error = report_consistency(engine)
            run_queries(engine, queries, show_why, quiet_error)
            continue
        print("unrecognized command (use '=FACTS', '?FACTS' or 'quit')", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Propositional-logic expert system")
    ap.add_argument("input_file", help="path to the rules/facts/queries file")
    ap.add_argument(
        "--interactive", "-i", action="store_true",
        help="after the initial run, allow changing facts and re-querying"
    )
    ap.add_argument(
        "--why", "-w", action="store_true",
        help="explain, for each query, which rules/facts determined the result"
    )
    ap.add_argument(
        "--graph", "-g", action="store_true",
        help="open a Tkinter window graphing every fact and rule; "
             "click a fact to trace its proof back to the initial facts"
    )
    args = ap.parse_args(argv)

    try:
        parsed = parse_input_file(args.input_file)
    except ExpertSystemError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    rules = expand_biconditionals(parsed.rules)
    engine = Engine(rules, parsed.initial_facts)
    quiet_error = report_consistency(engine)
    status = run_queries(engine, parsed.queries, args.why, quiet_error)
    if quiet_error:
        status = 1

    if args.graph:
        from expert_system.gui import launch_gui  # imports tkinter; keep it optional

        # Pass the un-expanded, one-rule-per-source-line list: that's what
        # the GUI's live rule editor displays and expands fresh on each edit.
        launch_gui(parsed.rules, parsed.initial_facts, parsed.queries, title=args.input_file)

    if args.interactive:
        return run_interactive(parsed, args.why)

    return status


if __name__ == "__main__":
    raise SystemExit(main())
