"""Regenerate tests/fork_freeze/dormant_modules.json during an upstream sync.

Lists every module under hermes_cli, gateway, tools, agent and cron that no live
module imports, with the SHA-256 of its bytes, and prints the modules that are
live only because live code loads them by name. Takes no arguments.

Static roots: pyproject.toml entry points, hermes_cli.main, run_agent,
acp_adapter.entry, gateway.run, model_tools, every top-level script, scripts/,
skills/, optional-skills/, plugins/ (scanned by hermes_cli/plugins.py and
providers/__init__.py) and providers/<name>.py (pkgutil scan in providers/).
By-name roots: the tools/*.py that discover_builtin_tools in tools/registry.py
imports (a top-level registry.register(...) call), and modules that non-Python
launchers start ("-m mod", "import mod", "from mod import" in ui-tui, apps,
docker, nix and shell scripts).
Static edges: every import statement anywhere in a file, relative imports
resolved, `from pkg import name` also importing pkg.name, parent packages.
By-name edges: a string constant equal to a module name, "module:attr",
"-m module" or a module's .py path (subprocess, runpy and spec_from_file_location
run files by path), and an f-string name passed to a dynamic import call, which loads
the modules it matches whose field values are string constants of the calling
module, or every module it matches when there are none.
"""
import ast
import hashlib
import json
import os
import posixpath
import re
import sys
import tomllib
import warnings
from collections import defaultdict

warnings.simplefilter("ignore", SyntaxWarning)  # invalid escapes in scanned sources are not our report
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tests", "fork_freeze", "dormant_modules.json")
FROZEN = ("hermes_cli", "gateway", "tools", "agent", "cron")
TOP_SKIP = {"website", "apps", "web", "ui-tui", "tests-js", "docs"}  # the draft's top-level skips
LAUNCHER_EXT = (".ts", ".tsx", ".js", ".mjs", ".cjs", ".sh", ".nix", ".ps1", ".cmd", ".bat")
LAUNCH_RX = re.compile(r"""(?:-m['"]?\s*,?\s*['"]?|\bimport\s+|\bfrom\s+)([A-Za-z_]\w*(?:\.\w+)+)""")
DYNAMIC_CALLS = {"import_module", "__import__", "run_module", "find_spec"}

if sys.argv[1:]:
    sys.exit("usage: python scripts/fork_dormant_modules.py (no arguments)")


def walk(top_skip):
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel = os.path.relpath(dirpath, ROOT)
        parts = [] if rel == "." else rel.split(os.sep)
        dirnames[:] = sorted(
            d for d in dirnames
            if not d.startswith(".") and d not in {"tests", "venv", "node_modules", "__pycache__"}
            and (parts or d not in top_skip)
        )
        for fn in sorted(filenames):
            yield parts, fn, os.path.join(dirpath, fn)


def rel(path):
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


def parents(name):
    out = []
    while "." in name:
        name = name.rpartition(".")[0]
        out.append(name)
    return out


mods = {}  # module name -> path; this script is no runtime code, so its own strings never count
for parts, fn, path in walk(TOP_SKIP):
    if fn.endswith(".py") and not fn.startswith("test_") and fn != "conftest.py" and path != os.path.abspath(__file__):
        name = ".".join(parts if fn == "__init__.py" else parts + [fn[:-3]])
        if name:
            mods[name] = path


by_path = {rel(p): m for m, p in mods.items()}


def named_in(s):
    """Modules a string names: its exact dotted name, "module:attr" or "-m module"."""
    m = re.fullmatch(r"([\w.]+):[\w.]+", s)
    return {c for c in [s, m and m.group(1)] + re.findall(r"-m\s+([\w.]+)", s) if c in mods}


def path_names(p, here):
    """Module files a .py path names: relative to the naming module's directory or the
    repository root, else every module file whose path ends with it (not a bare __init__.py)."""
    hits = {by_path.get(posixpath.normpath(posixpath.join(here, p))), by_path.get(posixpath.normpath(p))} - {None}
    if hits or p == "__init__.py":
        return hits
    return {m for r, m in by_path.items() if r == p or r.endswith("/" + p)}


def pattern_names(node, consts):
    """Modules an f-string name can match (a leading field may be a dotted name): those whose
    field values are string constants of the calling module, else every match (unsure)."""
    rx = "".join(
        re.escape(v.value) if isinstance(v, ast.Constant) else (r"([\w.]+)" if i == 0 else r"(\w+)")
        for i, v in enumerate(node.values)
    )
    hits = {m: g.groups() for m in mods if (g := re.fullmatch(rx, m))}
    return {m for m, fields in hits.items() if set(fields) <= consts} or set(hits)


def registers_tools(stmt):  # tools/registry.py _is_registry_register_call
    f = stmt.value.func if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) else None
    return isinstance(f, ast.Attribute) and f.attr == "register" and isinstance(f.value, ast.Name) and f.value.id == "registry"


static, byname = defaultdict(set), defaultdict(set)
byname_roots = defaultdict(set)  # module -> files that load it by name
for name, path in mods.items():
    with open(path, "rb") as fh:
        src = fh.read()
    try:
        tree = ast.parse(src, filename=path)
    except (SyntaxError, ValueError) as exc:
        sys.exit(f"cannot parse {rel(path)}: {exc}")
    pkg = name if path.endswith("__init__.py") else name.rpartition(".")[0]
    consts = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    r = rel(path)
    if (r.startswith("tools/") and r.count("/") == 1 and r[6:] not in {"__init__.py", "registry.py", "mcp_tool.py"}
            and b"registry" in src and b"register" in src and any(registers_tools(s) for s in tree.body)):
        byname_roots[name].add("tools/registry.py")
    chained = set()  # .py constants already read as the tail of a Path / "dir" / "x.py" chain
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            tail, n = [], node
            while isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div) and isinstance(n.right, ast.Constant) and isinstance(n.right.value, str):
                tail.insert(0, n.right.value)
                chained.add(id(n.right))
                n = n.left
            if tail and re.fullmatch(r"[\w./-]+\.py", "/".join(tail)):
                byname[name] |= path_names("/".join(tail), r.rpartition("/")[0])
        elif isinstance(node, ast.Import):
            targets = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                up = pkg.split(".") if pkg else []
                up = up[: len(up) - (node.level - 1)]
                base = ".".join(up + ([node.module] if node.module else []))
            targets = [base] + [f"{base}.{a.name}" if base else a.name for a in node.names]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            byname[name] |= named_in(node.value)
            if id(node) not in chained and re.fullmatch(r"[\w./-]+\.py", node.value):
                byname[name] |= path_names(node.value, r.rpartition("/")[0])
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.JoinedStr):
            f = node.func
            if (f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)) in DYNAMIC_CALLS:
                byname[name] |= pattern_names(node.args[0], consts)
        for t in targets:
            static[name] |= {c for c in [t] + parents(t) if c in mods}

with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
    project = tomllib.load(fh).get("project", {})
groups = [project.get("scripts", {}), project.get("gui-scripts", {})] + list(project.get("entry-points", {}).values())
roots = {v.partition(":")[0].strip() for g in groups for v in g.values()}
roots |= {"hermes_cli.main", "run_agent", "acp_adapter.entry", "gateway.run", "model_tools"}
for name, path in mods.items():
    r = rel(path)
    top, _, rest = r.partition("/")
    if not rest or top in {"scripts", "plugins", "skills", "optional-skills"}:
        roots.add(name)
    elif top == "providers" and "/" not in rest and not rest.startswith("_") and rest != "base.py":
        roots.add(name)

for parts, fn, path in walk({"website", "docs", "tests-js"}):
    if fn.endswith(".py") or ".test." in fn or ".spec." in fn:
        continue
    with open(path, "rb") as fh:
        head = fh.read(2)
    if fn.endswith(LAUNCHER_EXT) or fn == "Dockerfile" or ("." not in fn and head == b"#!"):
        with open(path, encoding="utf-8", errors="replace") as fh:
            for m in LAUNCH_RX.findall(fh.read()):
                if m in mods:
                    byname_roots[m].add(rel(path))


def reach(start, edge_maps):
    seen, stack = set(), [m for m in start if m in mods]
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        seen.add(m)
        stack.extend(p for p in parents(m) if p in mods)
        for edges in edge_maps:
            stack.extend(edges[m] - seen)
    return seen


live = reach(roots | set(byname_roots), (static, byname))
statically = reach(roots, (static,))
namers = defaultdict(set, {m: set(v) for m, v in byname_roots.items()})
importers = defaultdict(set)
for m in sorted(live):
    for t in byname[m] - {m}:
        namers[t].add(rel(mods[m]))
    for t in static[m]:
        importers[t].add(rel(mods[m]))

dormant = {}
for m in sorted(set(mods) - live):
    if rel(mods[m]).split("/")[0] in FROZEN:
        with open(mods[m], "rb") as fh:
            dormant[rel(mods[m])] = hashlib.sha256(fh.read()).hexdigest()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
    fh.write(json.dumps(dormant, indent=2, sort_keys=True) + "\n")

print(f"modules scanned={len(mods)} live={len(live)} dormant={len(mods) - len(live)} listed={len(dormant)}")
print("live only by name (module <- live code that names it; '(import)' = imported by a module live by name):")
for m in sorted(live - statically, key=lambda m: rel(mods[m])):
    if rel(mods[m]).split("/")[0] in FROZEN:
        why = sorted(namers[m]) or [f"(import) {p}" for p in sorted(importers[m])] or ["(package of a module live by name)"]
        print(f"  {rel(mods[m])} <- {', '.join(why)}")
