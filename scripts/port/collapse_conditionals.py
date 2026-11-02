#!/usr/bin/env python3
"""Collapse preprocessor conditionals to the branch the Linux build already takes.

Three-valued logic (True / False / Unknown) with the platform macros pinned to what Linux
defines. A condition that is definitely false loses its branch (its `#else` body is kept if there
is one); definitely true loses its guard lines; unknown terms are reported and left alone unless an
operand can be dropped without changing the outcome. Behaviour-preserving by construction: it only
removes code the compiler already removed.

Usage:
  python3 scripts/port/collapse_conditionals.py --tokens _WIN32,windows.h [--apply] FILE...
"""
import argparse
import re
import sys

TRUE_PLATFORM = {"__linux__", "__x86_64__", "__GNUC__", "_GNU_SOURCE", "__cplusplus",
                 "__GLIBC__", "__ELF__"}
FALSE_PLATFORM = {"_WIN32", "WIN32", "__APPLE__", "__CYGWIN__", "_MSC_VER", "__MINGW32__",
                  "__MINGW64__", "__MACH__", "TARGET_OS_MAC", "TARGET_OS_IPHONE",
                  "DS4_ROCM_BUILD", "DS4_ROCM"}

U, T, F = "U", True, False


def meet(a, b):
    return F if (a is F or b is F) else (U if (a is U or b is U) else T)


def join(a, b):
    return T if (a is T or b is T) else (U if (a is U or b is U) else F)


class Unparsable(Exception):
    pass


class Evaluator:
    def __init__(self, scope):
        self.scope = scope          # macros this tool is allowed to reason about

    def value(self, name):
        if name in TRUE_PLATFORM:
            return T
        if name in FALSE_PLATFORM:
            return F
        return U

    def evaluate(self, expr):
        toks = re.findall(r"defined\b|[A-Za-z_][A-Za-z0-9_]*|\d+[UL]*|\|\||&&|[()!&|]", expr)
        p = [0]

        def peek():
            return toks[p[0]] if p[0] < len(toks) else None

        def operand():
            t = peek()
            if t is None:
                raise Unparsable(expr)
            if t == "!":
                p[0] += 1
                v = operand()
                return F if v is T else (T if v is F else U)
            if t == "defined":
                p[0] += 1
                if peek() == "(":
                    p[0] += 1
                    name = toks[p[0]]
                    p[0] += 2
                    if toks[p[0] - 1] != ")":
                        raise Unparsable(expr)
                else:
                    name = toks[p[0]]
                    p[0] += 1
                if name == "defined":
                    raise Unparsable(expr)
                return self.value(name)
            if t == "(":
                p[0] += 1
                v = ors()
                if peek() != ")":
                    raise Unparsable(expr)
                p[0] += 1
                return v
            if re.fullmatch(r"\d+[UL]*", t):
                p[0] += 1
                return T if int(re.sub(r"\D", "", t)) else F
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", t):
                p[0] += 1
                return self.value(t)
            raise Unparsable(expr)

        def ands():
            v = operand()
            while peek() == "&&":
                p[0] += 1
                v = meet(v, operand())
            return v

        def ors():
            v = ands()
            while peek() == "||":
                p[0] += 1
                v = join(v, ands())
            return v

        v = ors()
        if p[0] != len(toks):
            raise Unparsable(expr)
        return v

    def drop_safe_operands(self, expr):
        """Remove operands that cannot change the outcome: a provably-true term inside `&&`, or a
        provably-false term inside `||`. Only for macros in scope."""
        out = expr
        for name in self.scope:
            v = self.value(name)
            if v is T:
                out = re.sub(r"\s*&&\s*!\s*defined\s*\(\s*" + name + r"\s*\)", "", out)
                out = re.sub(r"!\s*defined\s*\(\s*" + name + r"\s*\)\s*&&\s*", "", out)
                out = re.sub(r"\s*\|\|\s*defined\s*\(\s*" + name + r"\s*\)", "", out)
                out = re.sub(r"defined\s*\(\s*" + name + r"\s*\)\s*\|\|\s*", "", out)
            elif v is F:
                out = re.sub(r"\s*\|\|\s*defined\s*\(\s*" + name + r"\s*\)", "", out)
                out = re.sub(r"defined\s*\(\s*" + name + r"\s*\)\s*\|\|\s*", "", out)
                out = re.sub(r"\s*&&\s*!\s*defined\s*\(\s*" + name + r"\s*\)", "", out)
                out = re.sub(r"!\s*defined\s*\(\s*" + name + r"\s*\)\s*&&\s*", "", out)
            if re.fullmatch(r"\s*!?\s*defined\s*\(\s*" + name + r"\s*\)\s*", out):
                return ""
        return out.strip()

    def decide(self, expr):
        """(verdict, rewritten_expression) with verdict in DEAD/LIVE/MANUAL/UNPARSABLE."""
        try:
            v = self.evaluate(expr)
        except Unparsable:
            return ("UNPARSABLE", expr)
        verdict = {T: "LIVE", F: "DEAD", U: "MANUAL"}[v]
        if verdict != "MANUAL":
            return (verdict, expr)
        stripped = self.drop_safe_operands(expr)
        if not stripped.strip():
            return ("LIVE", "")
        try:
            v2 = self.evaluate(stripped)
        except Unparsable:
            v2 = U
        if any(name in stripped for name in self.scope) and v2 is not U:
            return ({T: "LIVE", F: "DEAD", U: "MANUAL"}[v2], stripped)
        if any(name in stripped for name in self.scope):
            return (verdict, expr) if v2 is U else (verdict, stripped)
        return (verdict, stripped)


def parse(path):
    lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
    root = {"children": [], "line": 0, "end": len(lines), "branches": []}
    stack = [root]
    pending = None
    for i, raw in enumerate(lines, 1):
        s = raw.strip()
        if pending is not None:
            pending[2] += " " + s
            if not s.endswith("\\"):
                node, kind, expr, bstart = pending[:4]
                if node["branches"]:
                    node["branches"][-1]["bend"] = bstart - 1
                node["branches"].append({"kind": kind, "expr": expr, "bstart": bstart,
                                         "bend": i - 1, "dir": i})
                pending = None
            continue
        if not s.startswith("#"):
            continue
        m = re.match(r"#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)", s)
        if not m:
            continue
        kind, rest = m.group(1), m.group(2)
        expr = ("defined(" + rest.strip() + ")") if kind == "ifdef" else \
               ("!defined(" + rest.strip() + ")") if kind == "ifndef" else rest
        if kind in ("if", "ifdef", "ifndef"):
            node = {"children": [], "line": i, "end": None, "branches": []}
            stack[-1]["children"].append(node)
            stack.append(node)
            if s.endswith("\\"):
                pending = [node, kind, expr, i + 1]
            else:
                node["branches"].append({"kind": kind, "expr": expr, "bstart": i + 1,
                                         "bend": None, "dir": i})
        elif kind in ("elif", "else"):
            cur = stack[-1]
            if cur["branches"]:
                cur["branches"][-1]["bend"] = i - 1
            if s.endswith("\\"):
                pending = [cur, kind, expr, i + 1]
            else:
                cur["branches"].append({"kind": kind, "expr": expr, "bstart": i + 1,
                                        "bend": None, "dir": i})
        elif kind == "endif":
            cur = stack.pop()
            if cur["branches"]:
                cur["branches"][-1]["bend"] = i - 1
            cur["end"] = i
    return lines, root


def plan_node(ev, scope, node, plans, blocked):
    branches = node["branches"]
    if not branches:
        return
    first = branches[0]
    in_scope = any(name in first["expr"] for name in scope)
    verdict, rewritten = ev.decide(first["expr"]) if in_scope else (None, first["expr"])
    if verdict is None:
        for b in branches[1:]:
            if not any(name in b["expr"] for name in scope):
                continue
            v2, _ = ev.decide(b["expr"])
            if v2 == "DEAD":
                plans.append((b["dir"], b["bend"], "drop-branch", b["expr"].strip()))
            else:
                blocked.append((b["dir"], f"{v2.lower()} elif/else branch", b["expr"].strip()))
        for child in node["children"]:
            plan_node(ev, scope, child, plans, blocked)
        return
    if verdict == "LIVE":
        plans.append((node["line"], node["line"], "drop-if-directive", first["expr"].strip()))
        plans.append((node["end"], node["end"], "drop-endif", first["expr"].strip()))
        for b in branches[1:]:
            plans.append((b["dir"], b["bend"], "drop-unreachable-branch", b["expr"].strip()))
        for child in node["children"]:
            plan_node(ev, scope, child, plans, blocked)
        return
    if verdict in ("UNPARSABLE", "MANUAL"):
        if rewritten != first["expr"].strip():
            plans.append((node["line"], node["line"], "rewrite-guard", rewritten))
        else:
            blocked.append((node["line"], verdict.lower(), first["expr"].strip()))
        for child in node["children"]:
            plan_node(ev, scope, child, plans, blocked)
        return
    live_at = None
    for idx, b in enumerate(branches[1:], 1):
        if b["kind"] == "else":
            live_at = idx
            break
        v2, _ = ev.decide(b["expr"])
        if v2 == "DEAD":
            continue
        if v2 == "LIVE":
            live_at = idx
            break
        blocked.append((node["line"], f"dead #if, later branch {v2.lower()}", b["expr"].strip()))
        return
    if live_at is None:
        plans.append((node["line"], node["end"], "drop-whole-node", first["expr"].strip()))
        return
    target = branches[live_at]
    if target["kind"] == "else":
        plans.append((node["line"], target["dir"], "drop-if-through-else", first["expr"].strip()))
    else:
        plans.append((node["line"], target["dir"] - 1, "drop-if-through-elif",
                      first["expr"].strip()))
        plans.append((target["dir"], target["dir"], "rewrite-guard", target["expr"].strip()))
    plans.append((node["end"], node["end"], "drop-endif", first["expr"].strip()))
    for b in branches[live_at + 1:]:
        plans.append((b["dir"], b["bend"], "drop-unreachable-branch", b["expr"].strip()))


def process(path, tokens, apply):
    ev = Evaluator(tokens)
    lines, root = parse(path)
    plans, blocked = [], []
    for child in root["children"]:
        plan_node(ev, tokens, child, plans, blocked)
    drops, rewrites = set(), {}
    for start, end, kind, payload in plans:
        if kind == "rewrite-guard":
            rewrites[start] = "#if " + payload
            continue
        drops.update(range(start, end + 1))
    out = [rewrites.get(i, line) for i, line in enumerate(lines, 1) if i not in drops]
    if apply and out != lines:
        open(path, "w", encoding="utf-8").write("\n".join(out))
    kinds = {}
    for _s, _e, kind, _x in plans:
        kinds[kind] = kinds.get(kind, 0) + 1
    return kinds, blocked


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", default="_WIN32,windows.h,Win32,DStorage,IoRing",
                    help="comma-separated macros/headers this sweep targets")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    scope = [t for t in args.tokens.split(",") if t]
    for path in args.files:
        kinds, blocked = process(path, scope, args.apply)
        print(f"{path}: {kinds or 'no edits'}, {len(blocked)} blocked")
        for line, why, expr in blocked[:40]:
            print(f"   BLOCKED {line}: {why} :: {expr[:80]}")


if __name__ == "__main__":
    main()
