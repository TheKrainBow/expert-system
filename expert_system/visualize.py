"""Builds a per-fact "resolution tree" (see --graph): the actual reasoning
chain for ONE queried fact, read right-to-left. The queried fact is the
rightmost node; every step needed to resolve it -- which rule fired, how its
condition's AND/OR/XOR/NOT combine, and each fact feeding into them -- unfolds
leftward, down to initial facts, defaults, or an unresolved cycle.

Two passes:
  build_resolution_tree(engine, root_fact) -- pure logic: walks
      Engine.proof recursively, and *replays* each firing rule's condition
      expression node by node (reusing the engine's own Kleene combinators,
      so it's not a guess) -- every AND/OR/XOR/NOT gets its own node with
      its own resolved value. No tkinter dependency.

  layout_tree(topology, node_box, ...) -- pure geometry: a node's distance
      from the root becomes its column (root = rightmost column, deeper =
      further left), each column is packed to its widest node, and a
      standard in-order tree layout picks the row.
"""

from __future__ import annotations

from itertools import count

from expert_system.ast_nodes import And, ASTNode, Fact, Not, Or, Rule, TriState, Xor
from expert_system.engine import Engine, Result, kleene_and, kleene_not, kleene_or, kleene_xor
from expert_system.errors import ContradictionError

MARGIN = 40

OP_SYMBOL = {And: "+", Or: "|", Xor: "^", Not: "!"}
OP_NAME = {And: "AND", Or: "OR", Xor: "XOR", Not: "NOT"}
OP_COMBINE = {And: kleene_and, Or: kleene_or, Xor: kleene_xor}


def _facts_with_polarity(node: ASTNode, negated: bool = False) -> list[tuple[str, bool]]:
    if isinstance(node, Fact):
        return [(node.name, negated)]
    if isinstance(node, Not):
        return _facts_with_polarity(node.operand, not negated)
    if isinstance(node, (And, Or, Xor)):
        return _facts_with_polarity(node.left, negated) + _facts_with_polarity(node.right, negated)
    raise TypeError(f"unknown AST node {node!r}")


def collect_all_fact_names(rules: list[Rule], initial_facts: set[str], queries: list[str]) -> list[str]:
    names: set[str] = set(initial_facts) | set(queries)
    for rule in rules:
        names |= {n for n, _ in _facts_with_polarity(rule.condition)}
        names |= {n for n, _ in _facts_with_polarity(rule.conclusion)}
    return sorted(names)


class _TreeBuilder:
    """One resolution tree is a proper tree, not a shared DAG: if the same
    fact is needed twice, it's expanded twice, each in its own branch --
    that's what "really splitted up, one graph per resolution" calls for.
    The only thing that must not recurse forever is an actual cycle *within
    a single branch* (A depends on B depends on A); that's guarded by
    `path`, the chain of fact names currently being expanded above us.
    """

    def __init__(self, engine: Engine):
        self.engine = engine
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple[str, str]] = []  # (cause_id, effect_id), left -> right
        self._counter = count()

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}{next(self._counter)}"

    def build(self, root_fact: str) -> str:
        root_id = self._expand_fact(root_fact, path=(), depth=0)
        self._prune_unreachable(root_id)
        return root_id

    def _prune_unreachable(self, root_id: str) -> None:
        """Rules/subtrees that were expanded (to decide whether they
        matched) but not kept as a `children` link are now orphaned --
        drop them so layout/drawing never sees them."""
        reachable: set[str] = set()
        stack = [root_id]
        while stack:
            nid = stack.pop()
            if nid in reachable:
                continue
            reachable.add(nid)
            stack.extend(self.nodes[nid]["children"])
        self.nodes = {nid: n for nid, n in self.nodes.items() if nid in reachable}
        self.edges = [(a, b) for a, b in self.edges if a in reachable and b in reachable]

    def _subtree_has_cycle_ref(self, node_id: str) -> bool:
        """Whether the (load-bearing part of the) subtree at `node_id`
        contains a cycle back-reference. Purely decorative siblings (see
        `_expand_conclusion_structural`) are skipped entirely -- they don't
        contribute to *why* `target` has its value, so a cycle inside one
        of them (e.g. the sibling's own resolution happens to loop back)
        must not disqualify an otherwise clean, non-circular explanation.
        """
        stack = [node_id]
        seen: set[str] = set()
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            n = self.nodes[nid]
            if n.get("informational"):
                continue
            if n.get("leaf") == "cycle_ref":
                return True
            stack.extend(n["children"])
        return False

    def _expand_informational_fact(self, name: str, depth: int) -> str:
        """A compact, terminal (no children) fact node: just its name and
        current value, for a sibling shown purely for context -- not its
        own recursive derivation, which would just duplicate whatever's
        already shown on the branch that's actually explaining `target`."""
        node_id = self._new_id("f")
        try:
            result = self.engine.evaluate(name)
            value, tainted, error = result.value.value, result.tainted, None
        except ContradictionError as exc:
            value, tainted, error = "Contradiction", False, str(exc)
        self.nodes[node_id] = {
            "kind": "fact", "id": node_id, "name": name, "value": value, "tainted": tainted,
            "initial": name in self.engine.initial_facts, "error": error,
            "proof_kind": None, "leaf": "initial" if name in self.engine.initial_facts else None,
            "depth": depth, "children": [], "informational": True,
        }
        return node_id

    def _expand_fact(self, name: str, path: tuple[str, ...], depth: int) -> str:
        node_id = self._new_id("f")

        if name in path:
            cached = self.engine.cache.get(name)
            self.nodes[node_id] = {
                "kind": "fact", "id": node_id, "name": name,
                "value": cached.value if cached else "Undetermined",
                "tainted": cached is None, "initial": False, "error": None,
                "proof_kind": None, "leaf": "cycle_ref", "depth": depth, "children": [],
            }
            return node_id

        try:
            result = self.engine.evaluate(name)
        except ContradictionError as exc:
            self.nodes[node_id] = {
                "kind": "fact", "id": node_id, "name": name, "value": "Contradiction",
                "tainted": False, "initial": name in self.engine.initial_facts,
                "error": str(exc), "proof_kind": None, "leaf": "error", "depth": depth, "children": [],
            }
            return node_id

        proof = self.engine.proof.get(name)
        node = {
            "kind": "fact", "id": node_id, "name": name,
            "value": result.value.value, "tainted": result.tainted,
            "initial": name in self.engine.initial_facts, "error": None,
            "proof_kind": proof.kind if proof else None,
            "leaf": None, "depth": depth, "children": [], "candidates": [],
        }
        self.nodes[node_id] = node

        if name in self.engine.initial_facts:
            # Declared true -- a complete, terminal explanation on its own.
            # Don't also drill into other rules that happen to mention it:
            # at best that's redundant, and at worst (e.g. one of those
            # rules loops back through an ancestor) it's actively misleading.
            node["leaf"] = "initial"
            return node_id

        # If one or more rules actually decided this fact's value, that's
        # the whole explanation -- show all of them (if two independently
        # prove it, both matter), but skip the ones that didn't contribute.
        # A non-contributing branch can otherwise be a deep, confusing
        # detour (e.g. a biconditional's reverse direction looping back
        # through the very fact we're explaining) that adds noise, not
        # signal, once a clean explanation already exists. Only when
        # *nothing* matches do we fall back to showing every rule that was
        # checked, since then "what was tried and why it failed" *is* the
        # explanation.
        all_rules = self.engine.rules_for(name)
        if not all_rules:
            node["leaf"] = "default"
            return node_id

        node["candidates"] = all_rules
        expansions = [
            self._expand_rule(rule, path + (name,), depth + 1, name, result) for rule in all_rules
        ]
        # A branch that loops back through an ancestor in *this* resolution
        # path (a cycle_ref somewhere inside it) only "matches" because it's
        # leaning on a value that's part of what we're currently explaining
        # -- circular, not independent evidence. Downgrade it so a clean,
        # non-circular explanation is preferred whenever one exists.
        for rid, matches in expansions:
            if matches and self._subtree_has_cycle_ref(rid):
                self.nodes[rid]["matches"] = False
        expansions = [(rid, self.nodes[rid]["matches"]) for rid, _ in expansions]
        matching_ids = [rid for rid, matches in expansions if matches]
        kept_ids = matching_ids if matching_ids else [rid for rid, _ in expansions]
        node["children"] = kept_ids
        for rid in kept_ids:
            self.edges.append((rid, node_id))

        if result.tainted:
            # Tainted + False: nothing outside the cycle ever proved this, so
            # it settles on the ordinary default. Tainted + Undetermined is
            # different -- genuine ambiguity that happens to also touch an
            # unresolved cycle -- and is left unlabeled here so the
            # `proof_kind == "ambiguous"` fallback below explains it instead.
            if result.value is not TriState.UNDETERMINED:
                node["leaf"] = "cycle"
        elif not matching_ids:
            node["leaf"] = "default_not_fired"
        return node_id

    @staticmethod
    def _conclusion_depth(node: ASTNode, target: str) -> int | None:
        """How many AND/OR/XOR/NOT layers sit between `node` and `target`'s
        occurrence in it -- 0 for a bare `Fact(target)`, None if `target`
        doesn't (uniquely) appear. Used to place the rule box the right
        number of hops *behind* the op chain that will wrap it, instead of
        overlapping the first (closest-to-target) op node."""
        if isinstance(node, Fact):
            return 0 if node.name == target else None
        if isinstance(node, Not):
            inner = _TreeBuilder._conclusion_depth(node.operand, target)
            return None if inner is None else inner + 1
        if isinstance(node, (And, Or, Xor)):
            left = _TreeBuilder._conclusion_depth(node.left, target)
            if left is not None:
                return left + 1
            right = _TreeBuilder._conclusion_depth(node.right, target)
            return None if right is None else right + 1
        raise TypeError(f"unknown AST node {node!r}")

    def _expand_rule(
        self, rule: Rule, path: tuple[str, ...], depth: int, target: str, final_result: Result
    ) -> tuple[str, bool]:
        """Returns (link_id, matches): `link_id` is the node that should be
        connected as `target`'s child -- the rule itself when its conclusion
        is a bare fact, or the top of the conclusion's AND/OR/XOR/NOT
        breakdown when it isn't (see `_expand_conclusion`). `matches` is
        whether this rule's vote is the untainted one that actually decided
        `target`'s value, which drives the fired/inactive GUI styling.

        The rule box sits `_conclusion_depth` hops behind `target` -- one
        past however many op nodes its conclusion needs to unwrap down to
        `target` -- so it never lands in the same column as one of them.
        """
        concl_offset = self._conclusion_depth(rule.conclusion, target) or 0
        rule_depth = depth + concl_offset
        rule_id = self._new_id("r")
        cond_id, cond_result = self._expand_expr(rule.condition, path, rule_depth + 1)
        self.edges.append((cond_id, rule_id))
        self.nodes[rule_id] = {
            "kind": "rule", "id": rule_id, "text": rule.source_text, "line": rule.line_no,
            "fired": False, "matches": False, "depth": rule_depth, "children": [cond_id],
        }

        if cond_result.value is TriState.FALSE:
            return rule_id, False

        self.nodes[rule_id]["fired"] = True

        if cond_result.value is TriState.UNDETERMINED:
            matches = (
                not cond_result.tainted
                and not final_result.tainted
                and final_result.value is TriState.UNDETERMINED
            )
            self.nodes[rule_id]["matches"] = matches
            return rule_id, matches

        if concl_offset == 0:
            # Conclusion is exactly `Fact(target)` -- connect directly, no
            # AND/OR/XOR/NOT wrapper needed.
            matches = not cond_result.tainted and not final_result.tainted and final_result.value is TriState.TRUE
            self.nodes[rule_id]["matches"] = matches
            return rule_id, matches

        # Condition holds: unfold the conclusion's own AND/OR/XOR/NOT
        # structure down to `target`, exactly like the condition side, so a
        # rule like "A+B=>Y+Z" or a sibling-based XOR/OR/AND deduction shows
        # every step instead of silently asserting the answer.
        concl_id, concl_res = self._expand_conclusion(rule.conclusion, target, True, rule_id, path, depth)
        if concl_res is None:
            return rule_id, False  # defensive; rules_for() guarantees target is in the conclusion

        matches = (
            not cond_result.tainted
            and not concl_res.tainted
            and not final_result.tainted
            and concl_res.value == final_result.value
        )
        self.nodes[rule_id]["matches"] = matches
        self.nodes[concl_id]["matches"] = matches
        return concl_id, matches

    def _expand_conclusion(
        self, node: ASTNode, target: str, assumed: bool, rule_id: str, path: tuple[str, ...], depth: int
    ) -> tuple[str | None, Result | None]:
        """Mirrors Engine._infer_conclusion's algebra, but builds a visible
        node for every step instead of only returning a value. Returns
        (link_id, Result): link_id is None when `node` doesn't (uniquely)
        involve `target` at all (mirrors _infer_conclusion returning None).

        `rule_id` is threaded through and only ever consumed once, at the
        Fact(target) leaf -- that's the point where "the rule fired" is
        actually what asserts the value; everything above it (NOT/AND/OR/
        XOR) is pure boolean algebra layered on top of that one premise.
        """
        if isinstance(node, Fact):
            if node.name != target:
                return None, None
            return rule_id, Result(TriState.TRUE if assumed else TriState.FALSE, False)

        if isinstance(node, Not):
            # `child_res` already IS target's inferred value -- the "not
            # assumed" flip above did that, by construction (Fact(target)
            # returns True/False from `assumed` directly). Unlike the
            # condition side, there's no separate "value of this
            # sub-expression" to compute: kleene_not(child_res) would give
            # the value of `!operand`, not of `target`, which is a
            # different thing whenever `target` sits directly under the
            # NOT (e.g. conclusion "!Z" for target=Z) -- so just pass it
            # through unchanged.
            child_id, child_res = self._expand_conclusion(
                node.operand, target, not assumed, rule_id, path, depth + 1
            )
            if child_res is None:
                return None, None
            if child_id is None:
                return None, child_res
            return self._new_op("!", "NOT", [child_id], child_res, depth), child_res

        if isinstance(node, And):
            if assumed:
                return self._expand_conclusion_structural(node, target, True, rule_id, path, depth, "+", "AND")
            return self._expand_conclusion_sibling(
                node, target, assumed, rule_id, path, depth, lambda a, o: (False if o else None), "+", "AND"
            )

        if isinstance(node, Or):
            if not assumed:
                return self._expand_conclusion_structural(node, target, False, rule_id, path, depth, "|", "OR")
            return self._expand_conclusion_sibling(
                node, target, assumed, rule_id, path, depth, lambda a, o: (None if o else True), "|", "OR"
            )

        if isinstance(node, Xor):
            return self._expand_conclusion_sibling(
                node, target, assumed, rule_id, path, depth, lambda a, o: a != o, "^", "XOR"
            )

        raise TypeError(f"unknown AST node {node!r}")

    def _expand_conclusion_structural(
        self,
        node: And | Or,
        target: str,
        assumed: bool,
        rule_id: str,
        path: tuple[str, ...],
        depth: int,
        symbol: str,
        opname: str,
    ) -> tuple[str | None, Result | None]:
        """AND-true / OR-false: `target` sits in exactly one operand, which
        is forced to `assumed` independent of the other operand's value."""
        l_id, l_res = self._expand_conclusion(node.left, target, assumed, rule_id, path, depth + 1)
        r_id, r_res = self._expand_conclusion(node.right, target, assumed, rule_id, path, depth + 1)
        chosen_id, chosen_res, other_side = (
            (l_id, l_res, node.right) if l_res is not None else (r_id, r_res, node.left)
        )
        if chosen_res is None:
            return None, None
        # The other side is forced too (both operands are, independent of
        # each other), even though its value isn't needed to pin `target`
        # down. Show it anyway -- otherwise this op node has only one
        # visible input, which reads as broken rather than "both sides are
        # forced; here's what the other one resolves to as well". Kept as a
        # compact terminal (name + value, not its own full derivation) and
        # marked informational: expanding it fully would just duplicate
        # whatever's already shown on `target`'s own side (often the very
        # same rule/rederivation), which is noise, not new information.
        # Placed at `depth - 1` -- the same column as `target` itself, since
        # both facts become known at the same moment (the rule firing).
        if isinstance(other_side, Fact):
            other_id = self._expand_informational_fact(other_side.name, depth - 1)
        else:
            other_id, _ = self._expand_expr(other_side, path, depth + 1)
            self.nodes[other_id]["informational"] = True
        children = [c for c in (chosen_id, other_id) if c is not None]
        op_id = self._new_op(symbol, opname, children, chosen_res, depth)
        return op_id, chosen_res

    def _expand_conclusion_sibling(
        self,
        node: And | Or | Xor,
        target: str,
        assumed: bool,
        rule_id: str,
        path: tuple[str, ...],
        depth: int,
        resolve,
        symbol: str,
        opname: str,
    ) -> tuple[str | None, Result | None]:
        """AND-false / OR-true / XOR: not structurally forced on its own,
        but solvable once the *other* operand's actual value is known --
        same algebra as Engine._infer_via_sibling, with the sibling itself
        shown as a real (recursively resolved) node, not a hidden lookup."""
        target_in_left = target in {n for n, _ in _facts_with_polarity(node.left)}
        target_in_right = target in {n for n, _ in _facts_with_polarity(node.right)}
        if target_in_left == target_in_right:
            return None, None
        target_side, other_side = (node.left, node.right) if target_in_left else (node.right, node.left)

        other_id, other_res = self._expand_expr(other_side, path, depth + 1)
        desired = None if other_res.value is TriState.UNDETERMINED else resolve(
            assumed, other_res.value is TriState.TRUE
        )

        if desired is None:
            # Genuinely ambiguous: the other side's value doesn't (yet) pin
            # `target` down. Still show `target`'s own side -- via rule_id,
            # the same way the resolved branch below does -- so the rule and
            # its condition stay visible instead of vanishing from the tree;
            # it's marked informational since we can't yet say what it
            # actually resolves to, only that this op is where the ambiguity
            # comes from.
            inner_id, _ = self._expand_conclusion(target_side, target, True, rule_id, path, depth + 1)
            if inner_id is not None:
                self.nodes[inner_id]["informational"] = True

            # If the other side is a bare fact, its own full derivation is
            # often just this same rule mirrored (e.g. "D=>A|B" is exactly
            # how both A and B end up ambiguous) -- expanding it in full
            # duplicates the whole condition chain and loops back here via a
            # cycle_ref. A compact name+value leaf makes the shared cause
            # ("this one ambiguous rule") read as one branch, not two near-
            # identical ones; its own full derivation is still one click
            # away from the facts list.
            display_other_id = other_id
            if isinstance(other_side, Fact):
                display_other_id = self._expand_informational_fact(other_side.name, depth + 1)

            res = Result(TriState.UNDETERMINED, other_res.tainted)
            children = [c for c in (display_other_id, inner_id) if c is not None]
            op_id = self._new_op(symbol, opname, children, res, depth)
            return op_id, res

        inner_id, inner_res = self._expand_conclusion(target_side, target, desired, rule_id, path, depth + 1)
        if inner_res is None:
            return other_id, None
        res = Result(inner_res.value, inner_res.tainted or other_res.tainted)
        children = [c for c in (other_id, inner_id) if c is not None]
        op_id = self._new_op(symbol, opname, children, res, depth)
        return op_id, res

    def _expand_expr(self, node: ASTNode, path: tuple[str, ...], depth: int) -> tuple[str, Result]:
        if isinstance(node, Fact):
            fact_id = self._expand_fact(node.name, path, depth)
            fnode = self.nodes[fact_id]
            if fnode["leaf"] == "error":
                result = Result(TriState.UNDETERMINED, tainted=True)
            else:
                result = Result(TriState(fnode["value"]), fnode["tainted"])
            return fact_id, result

        if isinstance(node, Not):
            child_id, child_res = self._expand_expr(node.operand, path, depth + 1)
            res = kleene_not(child_res)
            return self._new_op("!", "NOT", [child_id], res, depth), res

        if isinstance(node, (And, Or, Xor)):
            l_id, l_res = self._expand_expr(node.left, path, depth + 1)
            r_id, r_res = self._expand_expr(node.right, path, depth + 1)
            res = OP_COMBINE[type(node)](l_res, r_res)
            symbol, name = OP_SYMBOL[type(node)], OP_NAME[type(node)]
            return self._new_op(symbol, name, [l_id, r_id], res, depth), res

        raise TypeError(f"unknown AST node {node!r}")

    def _new_op(self, symbol: str, name: str, children: list[str], res: Result, depth: int) -> str:
        op_id = self._new_id("o")
        for c in children:
            self.edges.append((c, op_id))
        self.nodes[op_id] = {
            "kind": "op", "id": op_id, "symbol": symbol, "name": name,
            "value": res.value.value, "tainted": res.tainted, "depth": depth, "children": children,
        }
        return op_id


def build_resolution_tree(engine: Engine, root_fact: str) -> dict:
    builder = _TreeBuilder(engine)
    root_id = builder.build(root_fact)
    return {"root": root_id, "nodes": builder.nodes, "edges": builder.edges}


def layout_tree(
    topology: dict,
    node_box: dict[str, tuple[str, float]],
    row_gap: float = 56,
    column_gap: float = 40,
) -> dict:
    """Assign x/y to every node. `node_box[id] = (label, pixel_width)`.

    Depth from the root (0 = the queried fact) becomes the column; columns
    are walked deepest-first so the root lands on the right and each column
    is packed to its single widest node. Row (y) comes from a standard
    in-order tree layout over each node's `children`.
    """
    nodes = topology["nodes"]
    max_depth = max((n["depth"] for n in nodes.values()), default=0)

    columns: dict[int, list[str]] = {}
    for nid, n in nodes.items():
        columns.setdefault(n["depth"], []).append(nid)

    x_of: dict[str, float] = {}
    cursor = MARGIN
    for d in range(max_depth, -1, -1):
        ids = columns.get(d, [])
        col_width = max((node_box[nid][1] for nid in ids), default=0)
        cx = cursor + col_width / 2
        for nid in ids:
            x_of[nid] = cx
        cursor += col_width + column_gap
    total_width = cursor - column_gap + MARGIN if nodes else MARGIN * 2

    y_of: dict[str, float] = {}
    next_row = 0

    def assign_y(nid: str) -> float:
        nonlocal next_row
        children = nodes[nid]["children"]
        if not children:
            y = MARGIN + next_row * row_gap
            next_row += 1
            y_of[nid] = y
            return y

        # An "informational" child (see _expand_conclusion_structural, e.g.
        # showing B alongside A for "D=>A+B") is decorative, not a real
        # branch of the tree -- it never got a chance to be visited by the
        # normal row counter above, so place it explicitly just past the
        # center of its real siblings, instead of averaging it in as if it
        # were an equally-weighted subtree.
        real_children = [c for c in children if not nodes[c].get("informational")]
        informational_children = [c for c in children if nodes[c].get("informational")]

        if real_children:
            center = sum(assign_y(c) for c in real_children) / len(real_children)
        else:
            center = MARGIN + next_row * row_gap
            next_row += 1

        for i, c in enumerate(informational_children):
            # An informational node isn't always a bare leaf (e.g. the rule
            # box kept for display in an ambiguous sibling lookup still has
            # its own condition subtree beneath it) -- recurse first so its
            # descendants still get a row, then override just its own y with
            # the decorative offset.
            assign_y(c)
            y_of[c] = center + row_gap * 0.6 * (i + 1)

        y = center
        # A pure pass-through onto a single real child that itself fans out
        # to an informational sibling (target <- op <- [chain, sibling]):
        # nudge upward so target reads clearly above the op's center
        # instead of landing exactly level with it.
        if len(real_children) == 1 and any(
            nodes[c].get("informational") for c in nodes[real_children[0]]["children"]
        ):
            y = center - row_gap * 0.3

        y_of[nid] = y
        return y

    if nodes:
        assign_y(topology["root"])

    # The "nudge" above (and its cascading effect through several nested
    # informational branches) can push a y further up than any single leaf
    # row would suggest -- MARGIN + next_row*row_gap alone isn't a reliable
    # upper/lower bound anymore. Derive the height from where nodes actually
    # ended up, and shift everything down if the topmost one crept above
    # MARGIN (which would otherwise clip it off the top of the canvas).
    if y_of:
        min_y, max_y = min(y_of.values()), max(y_of.values())
        shift = MARGIN - min_y
        if shift:
            for nid in y_of:
                y_of[nid] += shift
        total_height = (max_y - min_y) + MARGIN * 2
    else:
        total_height = MARGIN * 2

    out_nodes = {}
    for nid, n in nodes.items():
        label, width = node_box[nid]
        out_nodes[nid] = {**n, "x": x_of[nid], "y": y_of[nid], "label": label, "width": width}

    return {
        "root": topology["root"],
        "nodes": out_nodes,
        "edges": topology["edges"],
        "width": total_width,
        "height": total_height,
    }
