"""GATE — no script references a global name that does not exist.

This test exists because of a specific escape. Stripping this folder down to the E60 path
removed a helper that `val_sweep`'s STOCK branch imported; the FUSED branch called the same
helper and was missed. Every one of the other 94 tests stayed green, because none of them
executes `val_sweep`'s fused branch -- it needs the 442M model, a checkpoint and a GPU. The
bug surfaced only in a real run, at the first sweep, AFTER the training steps: in the E60
schedule that is eight hours in.

A `NameError` on a branch nobody exercises is not a rare failure mode in this kind of edit --
it is the DEFAULT one, because the interpreter never looks at the line until it runs it. So
this is a static check: `symtable` reports which names each function reads from module scope,
and those must all be defined somewhere the module can see.

It is deliberately blunt (module-level bindings + imports + builtins, no flow analysis). It
catches exactly the class of defect above and nothing subtler, which is the point.

Run:  python -m pytest tests/test_no_undefined_names.py -q
"""
from __future__ import annotations

import ast
import builtins
import symtable
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
PACKAGE = Path(__file__).resolve().parents[1] / "src" / "rexg_data"


def _module_bindings(source: str) -> set[str]:
    """Everything the module binds at ANY level: module-scope names, imports (including the
    function-local ones this code uses heavily to defer heavy dependencies), comprehension
    and except targets. Deliberately over-inclusive -- a false negative here costs nothing,
    a false positive costs a wrong test failure."""
    names = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__debug__"}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.Global):
            names.update(node.names)
    return names


def undefined_globals(path: Path) -> list[str]:
    """Names any function in `path` reads from module scope that the module never binds."""
    source = path.read_text(encoding="utf-8")
    bound = _module_bindings(source)
    bad = []

    def walk(table: symtable.SymbolTable, where: str):
        for sym in table.get_symbols():
            if sym.is_global() and not sym.is_assigned() and sym.get_name() not in bound:
                bad.append(f"{path.name}:{where}: `{sym.get_name()}` is not defined anywhere "
                           f"in this module -- a NameError waiting for that branch to run")
        for child in table.get_children():
            walk(child, f"{where}.{child.get_name()}" if where else child.get_name())

    walk(symtable.symtable(source, str(path), "exec"), "")
    return bad


@pytest.mark.parametrize(
    "path",
    sorted(SCRIPTS.glob("*.py")) + sorted(PACKAGE.rglob("*.py")),
    ids=lambda p: p.name)
def test_no_undefined_globals(path: Path):
    assert not undefined_globals(path), "\n".join(undefined_globals(path))


def test_the_gate_fires_on_the_exact_bug_it_was_written_for(tmp_path):
    """The real escape, minimised: a helper's import removed from one branch while another
    branch still calls it. Shown to the gate, because a gate never shown its own failure case
    is an untested branch that prints ok."""
    broken = tmp_path / "val_sweep_like.py"
    broken.write_text(
        "def val_sweep(state_dict, fused):\n"
        "    if not fused:\n"
        "        return state_dict\n"
        "    return load_network_weights(state_dict, ())\n")
    found = undefined_globals(broken)
    assert found, "the gate did not fire on a call to a helper that does not exist"
    assert "load_network_weights" in found[0], found
    print(f"  broken case refused (verbatim): {found[0]}")

    fixed = tmp_path / "fixed.py"
    fixed.write_text(
        "def load_network_weights(sd, keys):\n"
        "    return sd\n"
        "def val_sweep(state_dict, fused):\n"
        "    return load_network_weights(state_dict, ()) if fused else state_dict\n")
    assert not undefined_globals(fixed)
