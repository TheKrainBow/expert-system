"""Native Tkinter visualization of a fact's resolution tree (see --graph).

Clicking a fact -- in the sidebar, or on a fact node anywhere in the drawn
tree -- rebuilds and redraws the canvas around *that* fact's resolution: the
fact itself on the right, and every AND/OR/XOR/NOT/rule/fact step that was
actually needed to resolve it, unfolding leftward down to initial facts,
defaults, or an unresolved cycle. It's a fresh tree per click, not a
highlight overlaid on one big shared graph.

The sidebar also hosts a live editor -- initial facts and add/remove rule --
that re-parses, rebuilds a fresh Engine, and redraws.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont

from expert_system.ast_nodes import Rule
from expert_system.engine import Engine
from expert_system.errors import ContradictionError, ExpertSystemError
from expert_system.input_file import expand_biconditionals
from expert_system.parser import parse_rule_line
from expert_system.visualize import build_resolution_tree, collect_all_fact_names, layout_tree

PALETTE = {
    "bg": "#f6f8f5",
    "panel": "#ecf1ea",
    "text": "#182420",
    "muted": "#5c6b62",
    "border": "#d2ddd0",
    "true": "#1f8a5f",
    "false": "#b23a34",
    "undet": "#a97a1f",
    "edge": "#b9c6b6",
    "rule_fill": "#e3eee7",
    "rule_border": "#9dc0af",
    "rule_fill_inactive": "#f6f8f5",
    "rule_border_inactive": "#c3ccc0",
    "rule_text_inactive": "#8b978e",
    "accent": "#0f766e",
    "entry_bg": "#ffffff",
}

FACT_DIAMETER = 44
FACT_R = FACT_DIAMETER / 2 - 2
RULE_H = 32
RULE_MIN_W = 80
RULE_MAX_W = 260
RULE_PAD = 22
OP_SIZE = 32
OP_R = OP_SIZE / 2 - 2

OP_BLURB = {
    "AND": "True only if both sides are True.",
    "OR": "True if at least one side is True.",
    "XOR": "True if exactly one side is True (exclusive or).",
    "NOT": "Flips True/False; stays Undetermined if the operand does.",
}


def color_for(node: dict) -> str:
    if node.get("error"):
        return PALETTE["false"]
    return {"True": PALETTE["true"], "False": PALETTE["false"]}.get(node["value"], PALETTE["undet"])


def explanation_for(fact: dict) -> str:
    if fact.get("error"):
        return fact["error"]
    leaf = fact.get("leaf")
    name = fact["name"]
    if leaf == "initial":
        return f"'{name}' is an initial fact."
    if leaf == "default":
        return f"No rule concludes '{name}' and it is not an initial fact -> False by default."
    if leaf == "default_not_fired":
        rules = fact.get("candidates", [])
        if len(rules) == 1:
            return f"'{rules[0].source_text}' concludes '{name}', but its condition is currently False -> False by default."
        listed = "; ".join(r.source_text for r in rules)
        return f"{len(rules)} rules could conclude '{name}' ({listed}), but none of their conditions currently hold -> False by default."
    if leaf == "cycle_ref":
        return f"'{name}' loops back to an ancestor already being resolved above -- click it to inspect '{name}' on its own."
    if leaf == "cycle":
        return (
            f"'{name}' only depends on itself through this cycle, with nothing outside it "
            f"to prove it True -> False by default (provisional: recomputed each time)."
        )
    kind = fact.get("proof_kind")
    if kind == "ambiguous":
        return f"A rule fires but its conclusion does not pin '{name}' down alone -> Undetermined."
    return "Traced back through its supporting rule to the facts (and initial facts) it needed."


class GraphApp:
    def __init__(self, rules: list[Rule], initial_facts: set[str], queries: list[str], title: str):
        # `rules` is the *display* list, one entry per source line (a <=> is
        # still a single Rule here); the edit panel shows and mutates this.
        # Each rebuild expands biconditionals fresh for the engine.
        self.display_rules: list[Rule] = list(rules)
        self.initial_facts: set[str] = set(initial_facts)
        self.queries: list[str] = list(queries)
        self.title_text = title

        self.selected: str | None = None
        self.data: dict = {}
        self.sidebar_rows: dict[str, tuple] = {}
        self.fact_values: dict[str, dict] = {}

        self.root = tk.Tk()
        self.root.title(f"Expert system — {title}")
        self.root.geometry("1280x820")
        self.root.configure(bg=PALETTE["bg"])
        self.root.bind("<Escape>", lambda e: self.root.destroy())

        self.mono = tkfont.Font(family="Courier New", size=11)
        self.mono_bold = tkfont.Font(family="Courier New", size=11, weight="bold")
        self.ui_font = tkfont.Font(family="Helvetica", size=10)
        self.ui_bold = tkfont.Font(family="Helvetica", size=10, weight="bold")

        self._build_sidebar_shell(title)
        self._build_canvas_area()
        self._rebuild()

    def run(self) -> None:
        self.root.mainloop()

    # ------------------------------------------------------------------
    # Sidebar: static shell (title/legend/editor) + dynamic fact list
    # ------------------------------------------------------------------

    def _build_sidebar_shell(self, title: str) -> None:
        sidebar = tk.Frame(self.root, width=280, bg=PALETTE["panel"])
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(
            sidebar, text=title, bg=PALETTE["panel"], fg=PALETTE["muted"],
            font=self.ui_bold, anchor="w", wraplength=250, justify="left",
        ).pack(fill="x", padx=14, pady=(14, 8))

        legend = tk.Frame(sidebar, bg=PALETTE["panel"])
        legend.pack(fill="x", padx=14, pady=(0, 10))
        for label, color in (
            ("True", PALETTE["true"]),
            ("False", PALETTE["false"]),
            ("Undetermined", PALETTE["undet"]),
        ):
            row = tk.Frame(legend, bg=PALETTE["panel"])
            row.pack(fill="x", pady=1)
            tk.Frame(row, width=9, height=9, bg=color).pack(side="left", padx=(0, 7))
            tk.Label(row, text=label, bg=PALETTE["panel"], fg=PALETTE["muted"], font=self.ui_font).pack(side="left")
        tk.Label(
            sidebar, text="filled circle = initial fact  ·  diamond = AND/OR/XOR/NOT step",
            bg=PALETTE["panel"], fg=PALETTE["muted"], font=("Helvetica", 8), anchor="w",
            wraplength=250, justify="left",
        ).pack(fill="x", padx=14, pady=(0, 6))

        self._build_editor(sidebar)
        tk.Frame(sidebar, height=1, bg=PALETTE["border"]).pack(fill="x", padx=14, pady=(6, 6))

        tk.Label(
            sidebar, text="FACTS (click to see its resolution)", bg=PALETTE["panel"], fg=PALETTE["muted"],
            font=("Helvetica", 9, "bold"), anchor="w", wraplength=250, justify="left",
        ).pack(fill="x", padx=14, pady=(2, 2))

        list_container = tk.Frame(sidebar, bg=PALETTE["panel"])
        list_container.pack(fill="both", expand=True)
        list_canvas = tk.Canvas(list_container, bg=PALETTE["panel"], highlightthickness=0)
        scrollbar = tk.Scrollbar(list_container, orient="vertical", command=list_canvas.yview)
        self.fact_list_inner = tk.Frame(list_canvas, bg=PALETTE["panel"])
        self.fact_list_inner.bind(
            "<Configure>", lambda e: list_canvas.configure(scrollregion=list_canvas.bbox("all"))
        )
        list_canvas.create_window((0, 0), window=self.fact_list_inner, anchor="nw")
        list_canvas.configure(yscrollcommand=scrollbar.set)
        list_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _build_editor(self, sidebar: tk.Frame) -> None:
        """Live editor: change initial facts, add a rule, remove a rule --
        no need to touch the input file or restart to try a hypothesis."""
        tk.Label(
            sidebar, text="INITIAL FACTS", bg=PALETTE["panel"], fg=PALETTE["muted"],
            font=("Helvetica", 9, "bold"), anchor="w",
        ).pack(fill="x", padx=14, pady=(4, 2))
        facts_row = tk.Frame(sidebar, bg=PALETTE["panel"])
        facts_row.pack(fill="x", padx=14)
        self.facts_entry = tk.Entry(facts_row, font=self.mono, bg=PALETTE["entry_bg"], fg=PALETTE["text"],
                                     relief="solid", bd=1)
        self.facts_entry.insert(0, "".join(sorted(self.initial_facts)))
        self.facts_entry.pack(side="left", fill="x", expand=True, ipady=2)
        self.facts_entry.bind("<Return>", lambda e: self.apply_facts())
        tk.Button(
            facts_row, text="Apply", command=self.apply_facts, relief="flat",
            bg=PALETTE["accent"], fg="white", bd=0, cursor="hand2", padx=8,
        ).pack(side="left", padx=(6, 0))
        self.facts_error = tk.Label(sidebar, text="", bg=PALETTE["panel"], fg=PALETTE["false"],
                                     font=("Helvetica", 9), anchor="w", wraplength=250, justify="left")
        self.facts_error.pack(fill="x", padx=14)

        tk.Label(
            sidebar, text="ADD RULE", bg=PALETTE["panel"], fg=PALETTE["muted"],
            font=("Helvetica", 9, "bold"), anchor="w",
        ).pack(fill="x", padx=14, pady=(8, 2))
        rule_row = tk.Frame(sidebar, bg=PALETTE["panel"])
        rule_row.pack(fill="x", padx=14)
        self.rule_entry = tk.Entry(rule_row, font=self.mono, bg=PALETTE["entry_bg"], fg=PALETTE["text"],
                                    relief="solid", bd=1)
        self.rule_entry.pack(side="left", fill="x", expand=True, ipady=2)
        self.rule_entry.bind("<Return>", lambda e: self.add_rule())
        tk.Button(
            rule_row, text="Add", command=self.add_rule, relief="flat",
            bg=PALETTE["accent"], fg="white", bd=0, cursor="hand2", padx=8,
        ).pack(side="left", padx=(6, 0))
        self.rule_error = tk.Label(sidebar, text="", bg=PALETTE["panel"], fg=PALETTE["false"],
                                    font=("Helvetica", 9), anchor="w", wraplength=250, justify="left")
        self.rule_error.pack(fill="x", padx=14)

        tk.Label(
            sidebar, text="RULES", bg=PALETTE["panel"], fg=PALETTE["muted"],
            font=("Helvetica", 9, "bold"), anchor="w",
        ).pack(fill="x", padx=14, pady=(8, 2))
        rules_container = tk.Frame(sidebar, bg=PALETTE["panel"])
        rules_container.pack(fill="x", padx=14)
        rules_canvas = tk.Canvas(rules_container, bg=PALETTE["panel"], highlightthickness=0, height=110)
        rules_scroll = tk.Scrollbar(rules_container, orient="vertical", command=rules_canvas.yview)
        self.rule_list_inner = tk.Frame(rules_canvas, bg=PALETTE["panel"])
        self.rule_list_inner.bind(
            "<Configure>", lambda e: rules_canvas.configure(scrollregion=rules_canvas.bbox("all"))
        )
        rules_canvas.create_window((0, 0), window=self.rule_list_inner, anchor="nw")
        rules_canvas.configure(yscrollcommand=rules_scroll.set)
        rules_canvas.pack(side="left", fill="both", expand=True)
        rules_scroll.pack(side="right", fill="y")

    def apply_facts(self) -> None:
        text = self.facts_entry.get().strip()
        letters = [c for c in text if not c.isspace()]
        bad = [c for c in letters if not (c.isalpha() and c.isupper())]
        if bad:
            self.facts_error.config(text=f"initial facts must be uppercase letters (got '{bad[0]}')")
            return
        self.facts_error.config(text="")
        self.initial_facts = set(letters)
        self._rebuild()

    def add_rule(self) -> None:
        text = self.rule_entry.get().strip()
        if not text:
            return
        try:
            rule = parse_rule_line(text, line_no=len(self.display_rules) + 1, raw_line=text)
        except ExpertSystemError as exc:
            self.rule_error.config(text=str(exc))
            return
        self.rule_error.config(text="")
        self.rule_entry.delete(0, "end")
        self.display_rules.append(rule)
        self._rebuild()

    def remove_rule(self, index: int) -> None:
        del self.display_rules[index]
        self._rebuild()

    def _refresh_rule_list(self) -> None:
        # Shown exactly as written -- a <=> rule stays a single clean <=>
        # line here, since this is the source of truth you're editing.
        # It's only split into its two => directions where that actually
        # matters: in the resolution graph, and only for whichever
        # direction was actually used to explain a fact (see
        # expand_biconditionals / _expand_rule) -- that's what makes the
        # graph read as a chain of plain implications without losing the
        # cleaner <=> shorthand here in the editor.
        for w in self.rule_list_inner.winfo_children():
            w.destroy()
        for idx, rule in enumerate(self.display_rules):
            self._add_rule_row(idx, rule.source_text)

    def _add_rule_row(self, index: int, text: str) -> None:
        row = tk.Frame(self.rule_list_inner, bg=PALETTE["panel"])
        row.pack(fill="x", pady=1)
        tk.Button(
            row, text="✕", command=lambda i=index: self.remove_rule(i), relief="flat",
            bg=PALETTE["panel"], fg=PALETTE["muted"], bd=0, cursor="hand2",
            font=("Helvetica", 8), padx=4,
        ).pack(side="left")
        tk.Label(
            row, text=text, bg=PALETTE["panel"], fg=PALETTE["text"],
            font=("Courier New", 9), anchor="w", wraplength=210, justify="left",
        ).pack(side="left", fill="x", expand=True)

    def _refresh_fact_list(self) -> None:
        for w in self.fact_list_inner.winfo_children():
            w.destroy()
        self.sidebar_rows = {}
        for name in sorted(self.fact_values):
            fv = self.fact_values[name]
            row = tk.Frame(self.fact_list_inner, bg=PALETTE["panel"], cursor="hand2")
            row.pack(fill="x", padx=6, pady=1)
            dot = tk.Frame(row, width=9, height=9, bg=color_for(fv))
            dot.pack(side="left", padx=(4, 6), pady=6)
            name_lbl = tk.Label(row, text=name, bg=PALETTE["panel"], fg=PALETTE["text"], font=self.mono_bold)
            name_lbl.pack(side="left")
            val_text = "contradiction" if fv["error"] else fv["value"]
            val_lbl = tk.Label(row, text=val_text, bg=PALETTE["panel"], fg=PALETTE["muted"], font=self.ui_font)
            val_lbl.pack(side="right", padx=(0, 6))
            for widget in (row, dot, name_lbl, val_lbl):
                widget.bind("<Button-1>", lambda e, n=name: self.show_resolution(n))
            self.sidebar_rows[name] = (row, dot, name_lbl, val_lbl)
        self._update_sidebar_selection()

    def _update_sidebar_selection(self) -> None:
        for name, (row, dot, name_lbl, val_lbl) in self.sidebar_rows.items():
            active = name == self.selected
            bg = PALETTE["accent"] if active else PALETTE["panel"]
            fg = "white" if active else PALETTE["text"]
            row.config(bg=bg)
            name_lbl.config(bg=bg, fg=fg)
            val_lbl.config(bg=bg, fg="white" if active else PALETTE["muted"])

    # ------------------------------------------------------------------
    # Canvas area (top bar + scrollable graph)
    # ------------------------------------------------------------------

    def _build_canvas_area(self) -> None:
        right = tk.Frame(self.root, bg=PALETTE["bg"])
        right.pack(side="left", fill="both", expand=True)

        topbar = tk.Frame(right, bg=PALETTE["panel"])
        topbar.pack(fill="x")
        self.topbar_title = tk.Label(
            topbar, text="Click a fact to see its resolution", bg=PALETTE["panel"],
            fg=PALETTE["text"], font=self.mono_bold, anchor="w",
        )
        self.topbar_title.pack(side="left", padx=14, pady=10)
        self.topbar_hint = tk.Label(
            topbar, text="", bg=PALETTE["panel"], fg=PALETTE["muted"], font=self.ui_font, anchor="w",
        )
        self.topbar_hint.pack(side="left", padx=(0, 10), fill="x", expand=True)
        tk.Frame(right, height=1, bg=PALETTE["border"]).pack(fill="x")

        canvas_frame = tk.Frame(right, bg=PALETTE["bg"])
        canvas_frame.pack(fill="both", expand=True)
        xscroll = tk.Scrollbar(canvas_frame, orient="horizontal")
        yscroll = tk.Scrollbar(canvas_frame, orient="vertical")
        self.canvas = tk.Canvas(
            canvas_frame, bg=PALETTE["bg"], highlightthickness=0,
            xscrollcommand=xscroll.set, yscrollcommand=yscroll.set,
        )
        xscroll.config(command=self.canvas.xview)
        yscroll.config(command=self.canvas.yview)
        yscroll.pack(side="right", fill="y")
        xscroll.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)

    # ------------------------------------------------------------------
    # Rebuild (engine/facts changed) vs. show_resolution (pick a fact)
    # ------------------------------------------------------------------

    def _rule_box(self, text: str) -> tuple[str, float]:
        """Fit `text` to a pixel-accurate rule box: (label, box_width)."""
        budget = RULE_MAX_W - RULE_PAD
        full_w = self.mono.measure(text)
        if full_w <= budget:
            return text, max(RULE_MIN_W, full_w + RULE_PAD)
        lo, hi, best = 0, len(text), "…"
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = text[:mid] + "…"
            if self.mono.measure(candidate) <= budget:
                best = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return best, RULE_MAX_W

    def _rebuild(self) -> None:
        engine_rules = expand_biconditionals(self.display_rules)
        self.engine = Engine(engine_rules, self.initial_facts)
        try:
            # Settles every fact once and re-derives each against the
            # others' final values, catching a contradiction that only
            # shows up once a cyclic fact has stopped being provisional --
            # see Engine.check_consistency. Any fact this raises for still
            # gets its own "Contradiction" entry below, from evaluate().
            self.engine.check_consistency()
        except ContradictionError:
            pass
        all_facts = collect_all_fact_names(engine_rules, self.initial_facts, self.queries)
        self.fact_values = {}
        for name in all_facts:
            try:
                res = self.engine.evaluate(name)
                self.fact_values[name] = {"value": res.value.value, "tainted": res.tainted, "error": None}
            except ContradictionError as exc:
                self.fact_values[name] = {"value": "Contradiction", "tainted": False, "error": str(exc)}

        self._refresh_rule_list()
        self._refresh_fact_list()

        if self.selected in self.fact_values:
            self.show_resolution(self.selected)
        elif self.queries:
            self.show_resolution(self.queries[0])
        else:
            self.selected = None
            self.data = {}
            self._draw_placeholder()

    def show_resolution(self, name: str) -> None:
        self.selected = name
        topology = build_resolution_tree(self.engine, name)
        node_box = {}
        for nid, n in topology["nodes"].items():
            if n["kind"] == "fact":
                node_box[nid] = (n["name"], FACT_DIAMETER)
            elif n["kind"] == "op":
                node_box[nid] = (n["symbol"], OP_SIZE)
            else:
                node_box[nid] = self._rule_box(n["text"])
        self.data = layout_tree(topology, node_box, row_gap=FACT_DIAMETER + 12)
        self._redraw_canvas()
        self._update_sidebar_selection()

        root = self.data["nodes"][self.data["root"]]
        label = "Contradiction" if root.get("error") else root["value"]
        self.topbar_title.config(text=f"{name} = {label}")
        self.topbar_hint.config(text=explanation_for(root))

    def _draw_placeholder(self) -> None:
        self.canvas.delete("all")
        self.canvas.configure(scrollregion=(0, 0, 800, 600))
        self.canvas.create_text(
            400, 280, text="No queries in this file -- click a fact on the left to see its resolution.",
            font=self.ui_font, fill=PALETTE["muted"],
        )

    def _redraw_canvas(self) -> None:
        self.canvas.delete("all")
        self.canvas.configure(scrollregion=(0, 0, self.data["width"], self.data["height"]))

        for cause_id, effect_id in self.data["edges"]:
            a, b = self.data["nodes"][cause_id], self.data["nodes"][effect_id]
            ax, ay = self._anchor_out(a)
            bx, by = self._anchor_in(b)
            self.canvas.create_line(
                ax, ay, bx, by, fill=PALETTE["edge"], width=1.6, arrow=tk.LAST, arrowshape=(8, 10, 3),
            )

        for n in self.data["nodes"].values():
            if n["kind"] == "rule":
                self._draw_rule(n)
            elif n["kind"] == "op":
                self._draw_op(n)
            else:
                self._draw_fact(n)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _anchor_out(self, node: dict) -> tuple[float, float]:
        if node["kind"] == "rule":
            return node["x"] + node["width"] / 2 + 6, node["y"] + RULE_H / 2
        if node["kind"] == "op":
            return node["x"] + OP_R + 4, node["y"]
        return node["x"] + FACT_R + 2, node["y"]

    def _anchor_in(self, node: dict) -> tuple[float, float]:
        if node["kind"] == "rule":
            return node["x"] - node["width"] / 2 - 6, node["y"] + RULE_H / 2
        if node["kind"] == "op":
            return node["x"] - OP_R - 4, node["y"]
        return node["x"] - FACT_R - 2, node["y"]

    def _bind_click(self, item: int, on_click) -> None:
        self.canvas.tag_bind(item, "<Button-1>", on_click)
        self.canvas.tag_bind(item, "<Enter>", lambda e: self.canvas.config(cursor="hand2"))
        self.canvas.tag_bind(item, "<Leave>", lambda e: self.canvas.config(cursor=""))

    def _draw_rule(self, r: dict) -> None:
        x, y, half_w = r["x"], r["y"], r["width"] / 2
        active = r.get("matches", True)
        fill = PALETTE["rule_fill"] if active else PALETTE["rule_fill_inactive"]
        border = PALETTE["rule_border"] if active else PALETTE["rule_border_inactive"]
        text_color = PALETTE["text"] if active else PALETTE["rule_text_inactive"]
        body = self.canvas.create_rectangle(
            x - half_w, y, x + half_w, y + RULE_H,
            fill=fill, outline=border, width=1.3, dash=None if active else (3, 2),
        )
        pin_l = self.canvas.create_rectangle(
            x - half_w - 6, y + RULE_H / 2 - 3, x - half_w, y + RULE_H / 2 + 3,
            fill=border, outline="",
        )
        pin_r = self.canvas.create_rectangle(
            x + half_w, y + RULE_H / 2 - 3, x + half_w + 6, y + RULE_H / 2 + 3,
            fill=border, outline="",
        )
        text_id = self.canvas.create_text(x, y + RULE_H / 2, text=r["label"], font=self.mono, fill=text_color)
        for item in (body, pin_l, pin_r, text_id):
            self._bind_click(item, lambda e, rr=r: self.show_rule_info(rr))

    def _draw_op(self, o: dict) -> None:
        x, y = o["x"], o["y"]
        color = color_for(o)
        pts = [x, y - OP_R, x + OP_R, y, x, y + OP_R, x - OP_R, y]
        shape = self.canvas.create_polygon(pts, fill=PALETTE["panel"], outline=color, width=2)
        text_id = self.canvas.create_text(x, y, text=o["symbol"], font=self.mono_bold, fill=PALETTE["text"])
        for item in (shape, text_id):
            self._bind_click(item, lambda e, oo=o: self.show_op_info(oo))

    def _draw_fact(self, f: dict) -> None:
        x, y = f["x"], f["y"]
        color = color_for(f)
        leaf = f.get("leaf")
        # Filled solid, same treatment as an initial fact, whenever this IS
        # the fact being resolved -- so it reads as the answer, standing out
        # in front of plain-outline context (like an informational sibling
        # shown alongside it, e.g. B next to A for "D=>A+B").
        is_root = f["id"] == self.data["root"]
        filled = leaf in ("initial", "error") or is_root
        oval = self.canvas.create_oval(
            x - FACT_R, y - FACT_R, x + FACT_R, y + FACT_R,
            fill=color if filled else PALETTE["panel"], outline=color, width=2,
            dash=(3, 2) if leaf in ("default", "cycle_ref") else None,
        )
        text_id = self.canvas.create_text(
            x, y, text=f["name"], font=self.mono_bold, fill="white" if filled else PALETTE["text"],
        )
        for item in (oval, text_id):
            self._bind_click(item, lambda e, n=f["name"]: self.show_resolution(n))

    # ------------------------------------------------------------------
    # Node info (rule / op click)
    # ------------------------------------------------------------------

    def show_rule_info(self, r: dict) -> None:
        if r.get("matches", True):
            status = ""
        elif not r.get("fired", True):
            status = "  (condition is False -- did not fire)"
        else:
            status = "  (fired, but did not decide the result)"
        self.topbar_title.config(text=f"line {r['line']}{status}")
        self.topbar_hint.config(text=r["text"])

    def show_op_info(self, o: dict) -> None:
        self.topbar_title.config(text=f"{o['name']} ({o['symbol']}) = {o['value']}")
        self.topbar_hint.config(text=OP_BLURB.get(o["name"], ""))


def launch_gui(rules: list[Rule], initial_facts: set[str], queries: list[str], title: str) -> None:
    app = GraphApp(rules, initial_facts, queries, title)
    app.run()
