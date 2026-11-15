"""Whether the kernels still count references to their arrays as they work.

Numba counts a reference to an array wherever one is handed on, and drops
the counting where it can see that it cancels. What it leaves is done for
every pixel, or every step, that a kernel works on, and costs more than the
work: left in, it was half the time of a stroke and two thirds of that of a
rub. The top of ``fusain/_engine.py`` says how the kernels are written so
that none is left; this tells whether that still holds::

    python tools/refcounts.py           # every kernel: what it still counts
    python tools/refcounts.py -v bear   # and where, in those named "bear..."

For each kernel, how many times it adds to a count and takes from one:
outside its loops (once a call) and inside them. The status is 1 if a kernel
counts anything inside a loop, or anything at all in one that is called for
every step of a stroke (:data:`EVERY_STEP`), which numba compiles into no
other; ``tests/test_refcounts.py`` runs it.

The kernels are compiled afresh, in a cache of their own (some twenty
seconds): numba keeps the LLVM of what it compiles, not of what it loads
from its cache.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from collections import Counter

#: The kernels called for every step of a stroke, a rub or a line of ink.
EVERY_STEP = ("bear", "rub_under", "roll_over")

_DEFINITION = re.compile(
    r'^define [^\n]*@"?(?P<name>_ZN[^"(]*)"?\(.*?\{\n(?P<body>.*?)^}',
    re.DOTALL | re.MULTILINE,
)
_LABEL = re.compile(r'^"?([^\s":]+)"?:')
_TARGET = re.compile(r'label %"?([^\s",\]]+)"?')
_COUNTING = re.compile(r"@NRT_(incref|decref)\(ptr (?:nonnull )?%?([^)]*)\)")


def _blocks(body: str) -> dict[str, list[str]]:
    """The basic blocks of a function in LLVM's text: the lines of each."""
    blocks: dict[str, list[str]] = {"entry": []}
    name = "entry"
    for line in body.splitlines():
        label = _LABEL.match(line)
        if label:
            name = label.group(1)
            blocks[name] = []
        else:
            blocks[name].append(line)
    return blocks


def _components(nodes: set[str], after: dict[str, set[str]]) -> list[set[str]]:
    """The strongly connected components of the blocks ``nodes``, where
    ``after`` says which blocks each can go to (Tarjan's algorithm, without
    recursion: a kernel has thousands of blocks)."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    found: list[set[str]] = []
    for root in nodes:
        if root in index:
            continue
        index[root] = low[root] = len(index)
        stack.append(root)
        on_stack.add(root)
        work = [(root, iter(sorted(after[root] & nodes)))]
        while work:
            node, rest = work[-1]
            child = next(rest, None)
            if child is None:
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[node])
                if low[node] == index[node]:
                    found.append(_pop_to(node, stack, on_stack))
            elif child not in index:
                index[child] = low[child] = len(index)
                stack.append(child)
                on_stack.add(child)
                work.append((child, iter(sorted(after[child] & nodes))))
            elif child in on_stack:
                low[node] = min(low[node], index[child])
    return found


def _pop_to(node: str, stack: list[str], on_stack: set[str]) -> set[str]:
    component: set[str] = set()
    while True:
        top = stack.pop()
        on_stack.discard(top)
        component.add(top)
        if top == node:
            return component


def loop_depths(blocks: dict[str, list[str]]) -> Counter[str]:
    """How many loops each block of a function is in."""
    after = {
        name: set(_TARGET.findall("\n".join(lines))) & set(blocks)
        for name, lines in blocks.items()
    }
    depth: Counter[str] = Counter()
    todo = [set(blocks)]
    while todo:
        nodes = todo.pop()
        for component in _components(nodes, after):
            first, *more = sorted(component)
            if not more and first not in after[first]:
                continue  # (a block that no path leads back to: not a loop)
            depth.update(component)
            # (the loops inside this one: what is left of it without the
            # blocks it is entered by)
            outside = nodes - component
            heads = {
                name
                for name in component
                if any(name in after[other] for other in outside)
            }
            todo.append(component - (heads or {first}))
    return depth


def counted(kernel) -> list[tuple[int, str, str, str]]:
    """The reference counting left in a compiled kernel: for each time it
    adds to a count (``incref``) or takes from one (``decref``), how deep
    in its loops that is, which of the two, of which array, and in which
    block of its LLVM."""
    name = kernel.py_func.__name__
    mangled = f"7_engine{len(name)}{name}B"
    left = []
    for signature in kernel.signatures:
        for definition in _DEFINITION.finditer(kernel.inspect_llvm(signature)):
            symbol = definition.group("name")
            if mangled in symbol and "cpython" not in symbol:
                left += _counted_in(definition.group("body"))
    return left


def _counted_in(body: str) -> list[tuple[int, str, str, str]]:
    blocks = _blocks(body)
    depth = loop_depths(blocks)
    return [
        (depth[block], found.group(1), found.group(2), block)
        for block, lines in blocks.items()
        for found in _COUNTING.finditer("\n".join(lines))
    ]


def kernels() -> dict[str, object]:
    """Every kernel of the engine, compiled by drawing with a stick, a tool,
    a pen and a stylus, and rendering. (What they call for every pixel is
    compiled into them: it is no kernel of its own.)"""
    from fusain import Sheet, Smudger, pastels, pencils, pens, tools
    from fusain import _engine as eng

    sheet = Sheet(64, 64, seed=1)
    sheet.shade((4, 4, 60, 60), pencils.HB)
    sheet.rub((4, 4, 60, 60), Smudger(tools.STUMP))
    sheet.shade((4, 4, 60, 60), pens.BLUE)
    sheet.indent([(4, 4), (60, 60)])
    sheet.render()
    # (a pencil over pastel lays a film on it, which shines)
    bed = Sheet(32, 32, seed=1)
    bed.shade((2, 2, 30, 30), pastels.soft("#f1e9d8"), 0.6, 8.0)
    bed.shade((2, 2, 30, 30), pencils.B2, 0.6, 4.0)
    bed.shine("graphite")
    return {
        name: kernel
        for name, kernel in vars(eng).items()
        if getattr(kernel, "signatures", None)
    }


def main(arguments: list[str]) -> int:
    with tempfile.TemporaryDirectory(prefix="fusain-refcounts-") as cache:
        # (numba reads it when it is first imported: by kernels())
        os.environ["NUMBA_CACHE_DIR"] = cache
        return report(arguments)


def report(arguments: list[str]) -> int:
    """Print what each kernel still counts; 1 if one counts as it works."""
    verbose = "-v" in arguments
    wanted = [word for word in arguments if word != "-v"]
    failed = False
    for name, kernel in kernels().items():
        if wanted and not any(name.startswith(word) for word in wanted):
            continue
        left = counted(kernel)
        inside = [entry for entry in left if entry[0] > 0]
        bad = bool(inside) or (name in EVERY_STEP and bool(left))
        failed = failed or bad
        told = f"{len(left) - len(inside)} a call, {len(inside)} inside its loops"
        print(f"{name:16s} {told if left else 'nothing':40s}{'  <--' if bad else ''}")
        if verbose:
            for depth, which, array, block in sorted(left):
                print(f"    {which} {array} ({depth} loops deep, in {block})")
    if failed:
        print("A kernel counts references as it works: see fusain/_engine.py.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
