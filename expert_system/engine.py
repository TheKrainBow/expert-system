"""Backward-chaining inference engine with 3-valued (Kleene) logic.

Facts default to False. A fact only becomes True by being an initial fact or
by some rule actually demonstrating it. `Undetermined` is reserved for
genuine ambiguity: a rule's condition holds, but its OR/XOR conclusion
doesn't say *which* side is the one that's true (e.g. "A is true, and if A
then B or C" leaves B and C each Undetermined).

Critical subtlety (cycles + cache)
-----------------------------------
`evaluate(fact)` walks the rules whose conclusion mentions `fact`, recursively
evaluating each rule's condition. If, while doing so, we come back around to a
fact that is already being evaluated (a cycle, e.g. A depends on B which
depends on A), that fact's value isn't known yet -- but "not known yet" is
exactly the situation the default-False rule is for, so we assume a
provisional `False` for it *without* caching it, and tag the result as
`tainted`. A cycle with nothing anchoring it from outside will keep
re-deriving this same provisional False forever, which is indistinguishable
from (and reports the same as) a real, settled False -- so pure unanchored
cycles correctly end up False, matching a fact with no rule at all.

Taintedness propagates through every combination (Kleene AND/OR/XOR/NOT and
the fact-vote combination). A composite result is only cached when it is
*not* tainted, i.e. when it did not depend on an unresolved cycle. This is
what lets an "anchor" outside the cycle (e.g. `C => A` with `C` true) resolve
`A` to a firm `True` even though a naive first pass through the `A <-> B`
cycle assumed both sides False.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from expert_system.ast_nodes import And, ASTNode, Fact, Not, Or, Rule, TriState, Xor
from expert_system.errors import ContradictionError

T, F, U = TriState.TRUE, TriState.FALSE, TriState.UNDETERMINED


@dataclass(frozen=True)
class Result:
    """A truth value paired with whether it depends on an unresolved cycle."""

    value: TriState
    tainted: bool


# ----------------------------------------------------------------------
# Kleene 3-valued combinators. Standalone (not methods) so the resolution
# -tree visualizer (expert_system.visualize) can replay the exact same
# per-operator combination the engine used, instead of guessing at it --
# there is exactly one place these truth tables are defined.
# ----------------------------------------------------------------------


def kleene_not(a: Result) -> Result:
    if a.value is T:
        return Result(F, a.tainted)
    if a.value is F:
        return Result(T, a.tainted)
    return Result(U, a.tainted)


def kleene_and(a: Result, b: Result) -> Result:
    if a.value is F and not a.tainted:
        return Result(F, False)
    if b.value is F and not b.tainted:
        return Result(F, False)
    if a.value is F or b.value is F:
        return Result(F, True)
    if a.value is T and b.value is T:
        return Result(T, a.tainted or b.tainted)
    return Result(U, a.tainted or b.tainted)


def kleene_or(a: Result, b: Result) -> Result:
    if a.value is T and not a.tainted:
        return Result(T, False)
    if b.value is T and not b.tainted:
        return Result(T, False)
    if a.value is T or b.value is T:
        return Result(T, True)
    if a.value is F and b.value is F:
        return Result(F, a.tainted or b.tainted)
    return Result(U, a.tainted or b.tainted)


def kleene_xor(a: Result, b: Result) -> Result:
    if a.value is U or b.value is U:
        # Either possible value of the undetermined side flips the result,
        # so it cannot be pinned down regardless of taint.
        return Result(U, a.tainted or b.tainted)
    value = T if (a.value is T) != (b.value is T) else F
    return Result(value, a.tainted or b.tainted)


@dataclass(frozen=True)
class Vote:
    """One rule's (or the initial-facts line's) opinion about a fact."""

    value: TriState
    tainted: bool
    rule: Rule | None  # None means "the initial facts line"

    @property
    def source(self) -> str:
        return "initial facts" if self.rule is None else self.rule.source_text


@dataclass(frozen=True)
class ProofNode:
    """Explains why a fact ended up with its final value.

    This picks out *one* representative vote -- used for quick messages
    (contradiction text, --why). The resolution-tree visualizer wants the
    *complete* picture (every rule that concluded this fact, whether or not
    it ended up deciding the value), so it doesn't rely on this; it walks
    Engine.rules_for()/infer_conclusion() itself. See expert_system.visualize.

    kind is one of:
      "initial"    -- declared true in the initial facts line
      "rule"       -- a single rule's condition/conclusion pins it down
      "ambiguous"  -- a rule fires but its conclusion doesn't pin this fact
                       down alone (e.g. an OR/XOR conclusion)
      "cycle"      -- only reachable through an unresolved cycle right now
      "default"    -- no rule (or none whose condition currently holds)
                       concludes this fact: false by default
    """

    kind: str
    rule: Rule | None


@dataclass(frozen=True)
class TraceEvent:
    """One line of the human-readable recursion trace (see --why).

    `depth` is how many fact-evaluations deep we are; rendering it with
    indentation is what makes the recursive structure legible.
    """

    depth: int
    text: str


class Engine:
    def __init__(self, rules: list[Rule], initial_facts: set[str]):
        self.rules = rules
        self.initial_facts = initial_facts
        self.cache: dict[str, TriState] = {}
        self.trace: list[TraceEvent] = []
        self.proof: dict[str, ProofNode] = {}
        # Pre-index rules by every fact name appearing in their conclusion,
        # so evaluate() doesn't rescan the whole ruleset for every fact.
        self._rules_by_fact: dict[str, list[Rule]] = {}
        for rule in self.rules:
            for name in self._fact_names(rule.conclusion):
                self._rules_by_fact.setdefault(name, []).append(rule)

    @staticmethod
    def _fact_names(node: ASTNode) -> set[str]:
        if isinstance(node, Fact):
            return {node.name}
        if isinstance(node, Not):
            return Engine._fact_names(node.operand)
        if isinstance(node, (And, Or, Xor)):
            return Engine._fact_names(node.left) | Engine._fact_names(node.right)
        raise TypeError(f"unknown AST node {node!r}")

    def rules_for(self, fact: str) -> list[Rule]:
        """Every rule whose conclusion mentions `fact` (fires or not)."""
        return list(self._rules_by_fact.get(fact, []))

    def check_consistency(self) -> None:
        """Catch contradictions that a single query can miss.

        `evaluate(fact)` only checks a rule's vote against whatever is
        already known *at the moment that rule is examined*. If fact Y only
        settles after fact X was already resolved and cached -- because Y
        was still tangled in an unresolved cycle the first time X's rules
        were checked -- a genuine conflict between Y's rule and X's cached
        value can slip past unnoticed. Example: `B => !A`, `B => A`, `A => B`,
        with A an initial fact. Evaluating A first needs B, which needs A
        back (a cycle) and provisionally reads False, so neither `B=>!A` nor
        `B=>A` casts a real vote; A resolves True from the initial fact
        alone. Only afterwards does querying B settle it to True using A's
        now-cached value -- but nothing goes back to recheck A against that,
        even though `B=>!A` with B now True directly contradicts it.

        This settles every fact mentioned anywhere in the ruleset once (so
        whatever's derivable gets cached), then repeatedly forces each one
        to be re-derived from scratch against everyone else's now-settled
        value, until a full round changes nothing. Re-deriving reuses
        evaluate()'s own vote collection, so a real conflict is caught by
        the same ContradictionError it always raises -- just now checked
        against final values instead of whatever was known mid-recursion.
        """
        facts: set[str] = set(self.initial_facts)
        for rule in self.rules:
            facts |= self._fact_names(rule.condition) | self._fact_names(rule.conclusion)
        ordered = sorted(facts)

        for name in ordered:
            self.evaluate(name)

        for _ in range(len(ordered) + 1):
            changed = False
            for name in ordered:
                before = self.cache.get(name)
                self.cache.pop(name, None)
                self.evaluate(name)
                if self.cache.get(name) != before:
                    changed = True
            if not changed:
                break

    def infer_conclusion(self, rule: Rule, target: str, stack: frozenset[str] = frozenset()) -> Result | None:
        """Public entry point for what `rule`'s conclusion, if asserted true,
        would mean for `target`. Used by the resolution-tree visualizer to
        show every rule that touches a fact, not just the one that won."""
        return self._infer_conclusion(rule.conclusion, target, True, rule, stack, 0)

    # ------------------------------------------------------------------
    # Fact resolution (backward chaining, with cache + cycle taint)
    # ------------------------------------------------------------------

    def evaluate(
        self, fact: str, stack: frozenset[str] = frozenset(), depth: int = 0
    ) -> Result:
        if fact in self.cache:
            self._log(depth, f"{fact} = {self.cache[fact]}  (already known)")
            return Result(self.cache[fact], tainted=False)

        if fact in stack:
            # Cycle: `fact` is still being decided further up this same
            # recursion, so treat it the same as any other not-yet-proven
            # fact -- provisional False, never cached. If a rule elsewhere
            # (outside this cycle) proves `fact` True, that untainted vote
            # will win once we get back to it; if nothing ever does, every
            # rule that needed this guess stays unproven too, and the whole
            # cycle settles on False -- exactly the ordinary default.
            self._log(
                depth,
                f"{fact} is already being evaluated higher up (cycle) "
                f"-> provisional False (default), not cached",
            )
            return Result(F, tainted=True)

        self._log(depth, f"{fact} ?")
        stack = stack | {fact}
        votes: list[Vote] = []

        if fact in self.initial_facts:
            self._log(depth + 1, f"'{fact}' is an initial fact -> True")
            votes.append(Vote(T, tainted=False, rule=None))

        rules = self._rules_by_fact.get(fact, [])
        if not rules and fact not in self.initial_facts:
            self._log(depth + 1, "no rule concludes this fact and it is not an initial fact")

        # If every rule below turns out to not fire only because one of its
        # conditions was itself a provisional cycle-guess, the "no rule
        # fired" default we fall back to is provisional too -- one of those
        # rules might really fire once the cycle it's waiting on resolves.
        pending_cycle = False

        for rule in rules:
            self._log(depth + 1, f"rule '{rule.source_text}': testing condition")
            cond = self._eval_expr(rule.condition, stack, depth + 2)
            if cond.value is F:
                if cond.tainted:
                    pending_cycle = True
                    self._log(
                        depth + 2,
                        f"condition is provisionally False (mid-cycle) -> rule doesn't fire "
                        f"*yet*, no vote for {fact} (but this could still change)",
                    )
                else:
                    self._log(
                        depth + 2,
                        f"condition is False -> rule does not fire, no vote for {fact}",
                    )
                continue  # rule does not fire, no vote
            if cond.value is U:
                self._log(
                    depth + 2,
                    f"condition is Undetermined{' (provisional)' if cond.tainted else ''} "
                    f"-> vote: {fact} = Undetermined",
                )
                votes.append(Vote(U, cond.tainted, rule=rule))
                continue
            # cond.value is TRUE: the conclusion is asserted true.
            inferred = self._infer_conclusion(rule.conclusion, fact, True, rule, stack, depth + 2)
            if inferred is None:
                # Fact appears in the conclusion but the connective (OR/XOR
                # asserted true) doesn't let us pin its individual value down.
                self._log(
                    depth + 2,
                    f"condition is True, but conclusion '{rule.conclusion!r}' does not "
                    f"pin {fact} down on its own -> vote: {fact} = Undetermined",
                )
                votes.append(Vote(U, cond.tainted, rule=rule))
            else:
                vote_tainted = cond.tainted or inferred.tainted
                self._log(
                    depth + 2,
                    f"condition is True{' (provisional)' if cond.tainted else ''} "
                    f"-> vote: {fact} = {inferred.value}"
                    + (" (provisional)" if inferred.tainted and not cond.tainted else ""),
                )
                votes.append(Vote(inferred.value, vote_tainted, rule=rule))

        value, tainted = self._combine_votes(fact, votes, pending_cycle)
        if not tainted:
            self.cache[fact] = value
        self.proof[fact] = self._select_proof(votes, value, tainted)
        self._log(
            depth,
            f"=> {fact} = {value}"
            + ("  (provisional: unresolved cycle, will not be cached)" if tainted else ""),
        )
        return Result(value, tainted)

    @staticmethod
    def _select_proof(votes: list[Vote], value: TriState, tainted: bool) -> ProofNode:
        """Pick one representative vote that explains the final outcome."""
        if tainted:
            # Tainted only ever pairs with False (the cyclic default) or
            # Undetermined (real ambiguity that also happens to involve an
            # unresolved cycle, e.g. two facts each defined only in terms of
            # the other's XOR/OR sibling) -- never True, since an untainted
            # True vote always wins outright before taint is even considered.
            # These need different explanations: one is "nothing proves this,
            # so False by default", the other is "still genuinely ambiguous".
            if value is U:
                tv = next((v for v in votes if v.value is U and v.tainted), None)
                return ProofNode("ambiguous", tv.rule if tv else None)
            tv = next((v for v in votes if v.tainted), None)
            return ProofNode("cycle", tv.rule if tv else None)
        if value is T:
            v = next(v for v in votes if v.value is T and not v.tainted)
            return ProofNode("initial" if v.rule is None else "rule", v.rule)
        if value is U:
            v = next(v for v in votes if v.value is U and not v.tainted)
            return ProofNode("ambiguous", v.rule)
        # value is F
        v = next((v for v in votes if v.value is F and not v.tainted), None)
        return ProofNode("rule", v.rule) if v is not None else ProofNode("default", None)

    def _log(self, depth: int, text: str) -> None:
        self.trace.append(TraceEvent(depth, text))

    def _combine_votes(
        self, fact: str, votes: list[Vote], pending_cycle: bool = False
    ) -> tuple[TriState, bool]:
        untainted_true = [v for v in votes if v.value is T and not v.tainted]
        untainted_false = [v for v in votes if v.value is F and not v.tainted]
        untainted_undetermined = any(v.value is U and not v.tainted for v in votes)
        any_tainted = any(v.tainted for v in votes)

        if untainted_true and untainted_false:
            raise ContradictionError(
                fact,
                f"'{untainted_true[0].source}' proves it True while "
                f"'{untainted_false[0].source}' proves it False",
            )

        if untainted_true:
            return T, False

        if untainted_false:
            # Solid proof of False is just as final as solid proof of True --
            # an unrelated vote elsewhere still being tainted (mid-cycle)
            # must not override a determination that's already settled.
            return F, False

        if any_tainted:
            # Cannot be sure yet: defer to a future, uncached re-evaluation.
            return U, True

        if untainted_undetermined:
            return U, False

        # No rule cast a firm vote: default False. Still provisional if that's
        # only because a rule was skipped over a cyclic guess that might yet
        # flip -- once nothing outside the cycle overrides it, it settles on
        # this same False for good.
        return F, pending_cycle

    # ------------------------------------------------------------------
    # Kleene 3-valued evaluation of a condition expression
    # ------------------------------------------------------------------

    def _eval_expr(self, node: ASTNode, stack: frozenset[str], depth: int) -> Result:
        # depth is only advanced when we recurse into evaluate() for a Fact
        # leaf, since that's the only point where "evaluating one fact"
        # nests inside "evaluating another fact". AND/OR/XOR/NOT are just
        # sub-steps of the same condition check, so they stay at `depth`.
        if isinstance(node, Fact):
            return self.evaluate(node.name, stack, depth)
        if isinstance(node, Not):
            return kleene_not(self._eval_expr(node.operand, stack, depth))
        if isinstance(node, And):
            return kleene_and(
                self._eval_expr(node.left, stack, depth), self._eval_expr(node.right, stack, depth)
            )
        if isinstance(node, Or):
            return kleene_or(
                self._eval_expr(node.left, stack, depth), self._eval_expr(node.right, stack, depth)
            )
        if isinstance(node, Xor):
            return kleene_xor(
                self._eval_expr(node.left, stack, depth), self._eval_expr(node.right, stack, depth)
            )
        raise TypeError(f"unknown AST node {node!r}")

    # ------------------------------------------------------------------
    # Structural inference: given a conclusion is asserted `assumed`, what
    # does that force `target` to be? Returns None if undetermined/unrelated.
    #
    # AND distributes over an assumed-True conclusion (both operands are
    # forced true), OR distributes over an assumed-False conclusion (both
    # forced false) -- purely structural, no need to know anything else.
    #
    # The other combinations (AND-false, OR-true, and XOR either way) are
    # NOT structurally forced on their own, but they're not automatically
    # ambiguous either: if `target` sits in one operand and the *other*
    # operand's actual value is already known (e.g. it doesn't mention
    # `target`, or defaults false with no rule of its own), plain algebra
    # can still pin `target` down. E.g. for "D => A^B" with A already known
    # True: A^B=True forces B=False. `_infer_via_sibling` does that lookup
    # via `_eval_expr` on the other operand, using the same `stack` so an
    # unresolved cycle there still comes back tainted/Undetermined rather
    # than silently assuming a value.
    # ------------------------------------------------------------------

    def _infer_conclusion(
        self, node: ASTNode, target: str, assumed: bool, rule: Rule, stack: frozenset[str], depth: int
    ) -> Result | None:
        """Returns None if `node` doesn't (uniquely) constrain `target` at
        all; otherwise a Result -- value T/F if pinned down, or U if it's
        genuinely ambiguous. `tainted` marks a Result that depends on an
        unresolved cycle elsewhere (via a sibling lookup) and so must not be
        treated as a stable, cacheable answer -- same discipline as the rest
        of the engine's taint handling.
        """
        if isinstance(node, Fact):
            if node.name != target:
                return None
            return Result(T if assumed else F, False)
        if isinstance(node, Not):
            return self._infer_conclusion(node.operand, target, not assumed, rule, stack, depth)
        if isinstance(node, And):
            if assumed:
                left = self._infer_conclusion(node.left, target, True, rule, stack, depth)
                right = self._infer_conclusion(node.right, target, True, rule, stack, depth)
                return self._merge_structural(left, right, target, rule)
            # AND=False: ambiguous unless the other side is confirmed True,
            # in which case this side must be False (the only way left).
            return self._infer_via_sibling(
                node, target, assumed, rule, stack, depth, lambda a, o: (False if o else None)
            )
        if isinstance(node, Or):
            if not assumed:
                left = self._infer_conclusion(node.left, target, False, rule, stack, depth)
                right = self._infer_conclusion(node.right, target, False, rule, stack, depth)
                return self._merge_structural(left, right, target, rule)
            # OR=True: ambiguous unless the other side is confirmed False,
            # in which case this side must be True (the only way left).
            return self._infer_via_sibling(
                node, target, assumed, rule, stack, depth, lambda a, o: (None if o else True)
            )
        if isinstance(node, Xor):
            # XOR always pins the target down once the other side is known:
            # target = assumed XOR other, regardless of what "other" is.
            return self._infer_via_sibling(
                node, target, assumed, rule, stack, depth, lambda a, o: a != o
            )
        raise TypeError(f"unknown AST node {node!r}")

    def _infer_via_sibling(
        self,
        node: And | Or | Xor,
        target: str,
        assumed: bool,
        rule: Rule,
        stack: frozenset[str],
        depth: int,
        resolve,
    ) -> Result | None:
        """`target` is one operand of `node`; try to pin it down using the
        *other* operand's actual evaluated value. `resolve(assumed, other:
        bool) -> desired_target: bool | None` encodes the operator's algebra;
        None means "still ambiguous even knowing the other side".

        Only applies when `target` appears in exactly one operand -- if it's
        in both (or neither, which shouldn't happen), that's too tangled to
        solve here and stays ambiguous.

        If the other operand's value is itself only a tainted/provisional
        guess (mid-cycle), we still use it, but mark whatever we conclude
        as tainted too -- an untainted, permanently-cached Undetermined
        here would be a *wrong* answer if the cycle later resolves, not a
        genuinely stable one.
        """
        target_in_left = target in self._fact_names(node.left)
        target_in_right = target in self._fact_names(node.right)
        if target_in_left == target_in_right:
            return None

        target_side, other_side = (node.left, node.right) if target_in_left else (node.right, node.left)
        other = self._eval_expr(other_side, stack, depth)
        if other.value is U:
            return Result(U, other.tainted)

        desired = resolve(assumed, other.value is T)
        if desired is None:
            return Result(U, other.tainted)

        inner = self._infer_conclusion(target_side, target, desired, rule, stack, depth)
        if inner is None:
            return None
        return Result(inner.value, inner.tainted or other.tainted)

    @staticmethod
    def _merge_structural(left: Result | None, right: Result | None, target: str, rule: Rule) -> Result | None:
        if left is None:
            return right
        if right is None:
            return left
        if left.value is U or right.value is U:
            return Result(U, left.tainted or right.tainted)
        if left.value is not right.value:
            if left.tainted or right.tainted:
                # The disagreement might just be a cycle artifact rather
                # than a real contradiction in the rule -- don't hard-error
                # over a value that could still change; stay ambiguous.
                return Result(U, True)
            raise ContradictionError(
                target,
                f"rule '{rule.source_text}' structurally asserts both True and "
                f"False for '{target}'",
            )
        return Result(left.value, left.tainted or right.tainted)
