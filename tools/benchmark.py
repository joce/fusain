"""How long the engine takes: shading, rubbing, hatching with a pen and
rendering a sheet of 1120 by 820 pixels. Each workload is timed in processor
time (``time.process_time``, which a busy machine inflates much less than the
wall clock) and the best of several runs is shown, with how far the runs
spread::

    python tools/benchmark.py                  # every workload, best of 7
    python tools/benchmark.py rub pen          # those whose name has "rub" or "pen"
    python tools/benchmark.py -n 15 render     # best of 15
    python tools/benchmark.py --save before.json

To compare two versions, give ``--against`` another checkout of fusain (a
clone or a ``git worktree``). Both versions run at once, each in a process of
its own that imports its own fusain and has compiled its kernels, and they
take each workload turn about, run by run, so that whatever else the machine
is doing falls on both alike. The ratio shown is this checkout's best over
the other's: below 1, this one is quicker::

    git worktree add ../fusain-base origin/main
    python tools/benchmark.py --against ../fusain-base

``--cpu NAME`` compiles the kernels of both for the processor NAME, with the
instructions it has and no others (``NUMBA_CPU_NAME``, and an empty
``NUMBA_CPU_FEATURES``: numba otherwise keeps those of this machine, and
``haswell`` alone still has AVX-512), and ``--cpu-against NAME`` the other
side's alone, so one checkout can be compared with itself compiled two ways.
On an AVX-512 machine, repeat a comparison with ``--cpu haswell``: a change
can gain there and lose without those wide registers, or the other way::

    python tools/benchmark.py --against . --cpu-against haswell

``--against`` may also name a file written by an earlier ``--save``; that
run's times are compared with these, one after the other, so noise on the
machine counts against the ratio.

Run it before and after a change meant to make drawing faster (the first run
after a change to ``fusain/_engine.py`` compiles the kernels, which is not
timed), and ``tools/fingerprint.py`` to see that the picture has not changed.
Where a stroke or a rub has become much slower for no change to what it
works out, ``tools/refcounts.py`` tells whether a kernel has started to
count references to its arrays.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import numpy as np

from fusain import Sheet, Smudger, charcoal, papers, pastels, pencils, pens, tools

W, H = 1120, 820
AREA = (0, 0, W, H)
RED, BLUE = "#c0392b", "#2456a6"


def _blank(paper=papers.CARTRIDGE) -> Sheet:
    return Sheet(W, H, paper=paper, seed=1)


def _charcoaled() -> Sheet:
    sheet = _blank(papers.CHARCOAL)
    sheet.shade(AREA, charcoal.VINE, 0.6, width=10)
    sheet.shade(AREA, charcoal.COMPRESSED, 0.4, width=8, angle=-40)
    return sheet


def _toned() -> Sheet:
    sheet = _blank()
    sheet.shade(AREA, pencils.B4, 0.6, width=8)
    return sheet


def _layered() -> Sheet:
    sheet = _toned()
    sheet.shade(AREA, pastels.soft(RED), 0.4, width=12, angle=-40)
    return sheet


def _short_strokes(sheet: Sheet, ends: str = "round") -> None:
    for y in np.linspace(10, H - 20, 40):
        for x in np.linspace(10, W - 50, 50):
            path = [(x, y), (x + 30.0, y + 9.0)]
            sheet.stroke(path, pencils.B2, 0.6, width=3, ends=ends)


#: Each workload: the sheet it starts from (made once, copied for each run)
#: and what is timed.
WORKLOADS: dict[str, tuple[Callable[[], Sheet], Callable[[Sheet], object]]] = {
    "shade HB, width 6": (_blank, lambda s: s.shade(AREA, pencils.HB)),
    "shade HB over a tone": (_toned, lambda s: s.shade(AREA, pencils.HB, angle=-40)),
    "shade soft pastel, width 24": (
        lambda: _blank(papers.SANDED),
        lambda s: s.shade(AREA, pastels.soft(RED), 0.6, width=24),
    ),
    "shade soft pastel over charcoal": (
        _charcoaled,
        lambda s: s.shade(AREA, pastels.soft(RED), 0.6, width=24),
    ),
    "shade vine charcoal on its side": (
        lambda: _blank(papers.CHARCOAL),
        lambda s: s.shade(AREA, charcoal.VINE, 0.5, width=6, tilt=1.0),
    ),
    "shade coloured pencil, firm": (
        _blank,
        lambda s: s.shade(AREA, pencils.coloured(BLUE), 0.9, width=6),
    ),
    "rub charcoal with a finger": (
        _charcoaled,
        lambda s: s.rub(AREA, tools.FINGER, 0.6),
    ),
    "rub graphite with a stump": (
        _toned,
        lambda s: s.rub(AREA, Smudger(tools.STUMP), 0.6),
    ),
    "rub charcoal with a kneaded eraser": (
        _charcoaled,
        lambda s: s.rub(AREA, tools.KNEADED, 0.6),
    ),
    "hatch with a ballpoint pen": (_blank, lambda s: s.shade(AREA, pens.BLUE, 0.6)),
    "pen over graphite": (_toned, lambda s: s.shade(AREA, pens.BLUE, 0.6)),
    "2000 short strokes": (_blank, _short_strokes),
    "2000 short strokes, flat ends": (_blank, lambda s: _short_strokes(s, "flat")),
    "render": (_layered, lambda s: s.render()),
    "reflectance": (_layered, lambda s: s.reflectance()),
}


def _warm_up() -> None:
    """Compile every kernel, or load it from numba's cache."""
    sheet = Sheet(64, 64, seed=1)
    sheet.shade((4, 4, 60, 60), pencils.HB)
    sheet.shade((4, 4, 60, 60), pencils.HB, ends="flat")
    sheet.rub((4, 4, 60, 60), tools.FINGER)
    sheet.shade((4, 4, 60, 60), pens.BLUE)
    sheet.indent([(4, 4), (60, 60)])
    sheet.render()


# ----------------------------------------------------------------------
# The worker: a process that times workloads, one at a time, on request.
# ----------------------------------------------------------------------
def _send(message: dict) -> None:
    print(json.dumps(message), flush=True)


def _serve() -> int:
    """Time the workloads whose names arrive on standard input, in processor
    time, and answer each with its seconds. Only the last workload's sheet is
    kept (a sheet is some tens of megabytes)."""
    import fusain

    _warm_up()
    _send({"fusain": fusain.__file__})
    kept: tuple[str, Sheet] | None = None
    for line in sys.stdin:
        name = line.strip()
        start, work = WORKLOADS[name]
        if kept is None or kept[0] != name:
            kept = (name, start())
        sheet = kept[1].copy()
        gc.collect()
        began = time.process_time()
        work(sheet)
        _send({"seconds": time.process_time() - began})
    return 0


class Worker:
    """A worker process importing the fusain of one checkout, its kernels
    compiled for ``cpu`` (numba's ``NUMBA_CPU_NAME``, with that processor's
    own instructions; None for this machine, or what the environment already
    says)."""

    def __init__(self, root: Path, cpu: str | None) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(root), *filter(None, [env.get("PYTHONPATH")])]
        )
        if cpu:
            env["NUMBA_CPU_NAME"] = cpu
            env["NUMBA_CPU_FEATURES"] = ""  # (the named processor's own)
        self.root = root
        self.cpu = env.get("NUMBA_CPU_NAME", "native")
        self.process = subprocess.Popen(
            [sys.executable, __file__, "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=env,
        )
        where = Path(self._answer()["fusain"]).resolve()
        if root not in where.parents:
            raise SystemExit(f"{root}: imports fusain from {where}, not from there")

    def _answer(self) -> dict:
        line = self.process.stdout.readline()  # type: ignore[union-attr]
        if not line:
            raise SystemExit(f"{self.root}: the worker has stopped")
        return json.loads(line)

    def time(self, name: str) -> float:
        """Processor time of one run of workload ``name``, in seconds."""
        self.process.stdin.write(name + "\n")  # type: ignore[union-attr]
        self.process.stdin.flush()  # type: ignore[union-attr]
        return self._answer()["seconds"]

    def close(self) -> None:
        self.process.stdin.close()  # type: ignore[union-attr]
        self.process.wait()


# ----------------------------------------------------------------------
# The measuring and the report.
# ----------------------------------------------------------------------
def _measure(
    workers: Sequence[Worker], names: Sequence[str], runs: int
) -> Iterator[tuple[str, list[list[float]]]]:
    """For each workload, the times of every worker. The workers take their
    runs turn about, and the order they go in is reversed on every other
    round, so that neither always follows the other."""
    for name in names:
        times: list[list[float]] = [[] for _ in workers]
        for run in range(runs):
            order = range(len(workers))
            for i in reversed(order) if run % 2 else order:
                times[i].append(workers[i].time(name))
        yield name, times


def _cell(times: Sequence[float]) -> str:
    """The best time, and how far the runs spread above it."""
    best = min(times)
    return f"{best:7.3f} s +{100 * (max(times) / best - 1):3.0f}%"


def _row(name: str, *sides: Sequence[float]) -> str:
    row = f"{name:34s}" + "".join(f" {_cell(times):>16s}" for times in sides)
    if len(sides) == 2:
        row += f" {min(sides[0]) / min(sides[1]):8.3f}"
    return row


def _describe(label: str, root: Path | str, cpu: str | None) -> str:
    return f"{label:8s} {root}" + (f"  (cpu {cpu})" if cpu else "")


def _arguments(arguments: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Time the engine's workloads (processor time, best of N).",
        epilog="See the top of this file for how to compare two versions.",
    )
    parser.add_argument("words", nargs="*", help="only workloads whose name has one")
    parser.add_argument("-n", "--runs", type=int, default=7, help="runs of each (7)")
    parser.add_argument(
        "--against",
        metavar="CHECKOUT_OR_FILE",
        help="another checkout of fusain to run alongside this one, run by run, "
        "or a file written by --save",
    )
    parser.add_argument(
        "--cpu", metavar="NAME", help="compile the kernels for this CPU (e.g. haswell)"
    )
    parser.add_argument(
        "--cpu-against",
        metavar="NAME",
        help="the CPU for the other checkout alone (default: --cpu)",
    )
    parser.add_argument("--save", metavar="FILE", help="write the times to a file")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(arguments)


def _compare(args: argparse.Namespace, names: list[str]) -> dict[str, list[float]]:
    """Run, print the rows, and return this checkout's times."""
    here = Path(__file__).resolve().parent.parent
    other = Path(args.against).resolve() if args.against else None
    saved = None
    if other is not None and other.is_file():
        saved = json.loads(other.read_text())["times"]
        other = None
    workers = [Worker(here, args.cpu)]
    if other is not None:
        workers.append(Worker(other, args.cpu_against or args.cpu))
    print(_describe("here", here, workers[0].cpu))
    if other is not None:
        print(_describe("against", other, workers[1].cpu))
    if saved is not None:
        print(_describe("against", args.against, None))
    print(f"best of {args.runs}, processor time; + is how far the runs spread\n")
    results: dict[str, list[float]] = {}
    ratios: list[float] = []
    try:
        for name, times in _measure(workers, names, args.runs):
            results[name] = times[0]
            sides = times if saved is None else [times[0], saved.get(name, [])]
            sides = [side for side in sides if side]
            print(_row(name, *sides), flush=True)
            if len(sides) == 2:
                ratios.append(min(sides[0]) / min(sides[1]))
    finally:
        for worker in workers:
            worker.close()
    if ratios:
        mean = math.exp(statistics.fmean(map(math.log, ratios)))
        print(f"\n{'geometric mean of the ratios':34s} {mean:{16 * 2 + 3}.3f}")
    return results


def main(arguments: Sequence[str]) -> int:
    args = _arguments(arguments)
    if args.worker:
        return _serve()
    names = [
        name
        for name in WORKLOADS
        if not args.words or any(word in name for word in args.words)
    ]
    results = _compare(args, names)
    if args.save:
        saved = {"runs": args.runs, "cpu": args.cpu, "times": results}
        Path(args.save).write_text(json.dumps(saved, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
