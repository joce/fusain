"""The fingerprint of what the engine draws (``tools/fingerprint.py``): the
drawings of its suite still give the hashes recorded for them, so a change
that is not meant to alter the picture has not."""

import functools
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from fusain import Sheet, Smudger, Stick, pencils, pens, tools

TOOL = Path(__file__).parent.parent / "tools" / "fingerprint.py"


def _tool():
    spec = importlib.util.spec_from_file_location("fingerprint", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fp = _tool()
RECORDED = fp.load()


@functools.cache
def _same_arithmetic() -> bool:
    return fp.arithmetic() == RECORDED["arithmetic"]


def test_the_suite_is_the_recorded_one():
    assert {name for name, *_ in fp.runs()} == set(RECORDED["marks"])
    # (the thorough suite draws the recorded one too, and much more)
    thorough = [name for name, *_ in fp.runs(thorough=True)]
    assert set(RECORDED["marks"]) < set(thorough)
    assert len(thorough) > 10 * len(RECORDED["marks"])


@pytest.mark.parametrize(
    ("name", "scene", "paper", "seed"), list(fp.runs()), ids=lambda v: str(v)[:40]
)
def test_a_drawing_gives_the_recorded_hashes(name, scene, paper, seed):
    marks = fp.draw(scene, paper, seed)
    differing = fp.differences({name: marks}, {name: RECORDED["marks"][name]})
    if differing and not _same_arithmetic():
        pytest.skip(
            "this machine's arithmetic is not that of the recorded hashes "
            f"({RECORDED['recorded on']})"
        )
    assert not differing, (
        "the picture has changed (if it was meant to, record it again: "
        "python tools/fingerprint.py --record)"
    )


def test_every_mark_is_there():
    """Each drawing is looked at as a whole at its end: all a sheet holds,
    what it shows, and the numbers its calls returned."""
    held = ("tooth", "press", "dent", "cover", "held", "under", "loose")
    for name, marks in RECORDED["marks"].items():
        for key in (*held, "_fresh", "bonded", "_rng", "ground"):
            assert f"sheet.{key}" in marks, (name, key)
        for key in ("reflectance", "shine", "image", "raking"):
            assert f"sheet:{key}" in marks, (name, key)
        assert "returned" in marks
    inked = RECORDED["marks"]["ballpoint/printer paper/11"]
    assert {"sheet.ink", "sheet.inked", "sheet:overhead"} <= set(inked)
    carried = RECORDED["marks"]["dust/charcoal paper/4"]
    assert {"finger._carried", "stump:end._carried"} <= set(carried)
    worn = RECORDED["marks"]["sticks/Ingres/3"]
    assert {"pencil.worn", "pencil:end.point", "twig:end.worn"} <= set(worn)


def test_a_trace_hashes_all_that_a_sheet_a_stick_and_a_smudger_hold():
    sheet = Sheet(24, 16, seed=1)
    sheet.stroke([(4, 8), (20, 8)], pens.BLUE, 0.8)
    stick, stump = Stick(pencils.B2), Smudger(tools.STUMP)
    trace = fp.Trace()
    for name, thing in (("sheet", sheet), ("stick", stick), ("stump", stump)):
        trace.state(name, thing)
        arrays = {k for k, v in vars(thing).items() if isinstance(v, np.ndarray)}
        numbers = {k for k, v in vars(thing).items() if isinstance(v, int | float)}
        assert {f"{name}.{key}" for key in arrays | numbers} <= set(trace.marks)
    assert {"sheet.ink", "sheet.inked", "sheet._rng", "stump._carried"} <= set(
        trace.marks
    )
    assert {"stick.worn", "stick.point", "stick.lead"} <= set(trace.marks)
    # what is drawn afterwards shows in the marks of what it changed
    before = dict(trace.marks)
    trace.got(sheet.stroke([(4, 4), (20, 12)], stick, 0.7))
    for name, thing in (("sheet", sheet), ("stick", stick), ("stump", stump)):
        trace.state(name, thing)
    changed = {key for key, mark in trace.marks.items() if before[key] != mark}
    assert {"sheet.held", "sheet.cover", "sheet._rng", "stick.worn"} <= changed
    assert not {"sheet.tooth", "sheet.ink", "stump._carried", "stick.lead"} & changed
    assert trace.close()["returned"] != fp.Trace().close()["returned"]


def test_a_digest_sees_one_bit_a_type_and_a_shape():
    values = np.linspace(0.0, 1.0, 12, dtype=np.float32)
    nudged = values.copy()
    nudged[5] = np.nextafter(nudged[5], np.float32(2.0))
    marks = {
        fp.digest(values),
        fp.digest(nudged),
        fp.digest(values.astype(np.float64)),
        fp.digest(values.reshape(3, 4)),
    }
    assert len(marks) == 4
    assert fp.digest(values) == fp.digest(values[::-1][::-1].copy())


def test_differences_name_what_differs():
    old = {"a": {"x": "1", "y": "2"}, "b": {"x": "3"}}
    assert fp.differences(old, old) == []
    new = {"a": {"x": "1", "y": "5", "z": "6"}, "c": {"x": "3"}}
    assert fp.differences(new, old) == [
        "a: y differs",
        "a: z differs",
        "b: not drawn",
        "c: not recorded",
    ]


def test_the_tool_compares_and_saves(tmp_path, monkeypatch, capsys):
    marks = RECORDED["marks"]
    monkeypatch.setattr(fp, "fingerprint", lambda thorough=False: marks)
    monkeypatch.setattr(fp, "arithmetic", lambda: RECORDED["arithmetic"])
    assert fp.main([]) == 0
    assert "identical" in capsys.readouterr().out
    saved = tmp_path / "saved.json"
    assert fp.main(["--save", str(saved)]) == 0
    assert json.loads(saved.read_text(encoding="utf-8"))["marks"] == marks
    assert fp.main(["--thorough", "--against", str(saved)]) == 0
    # one bit of one array, elsewhere: it says which, and fails
    name = next(iter(marks))
    changed = {**marks, name: {**marks[name], "sheet.held": "0" * 16}}
    monkeypatch.setattr(fp, "fingerprint", lambda thorough=False: changed)
    monkeypatch.setattr(fp, "arithmetic", lambda: "another")
    capsys.readouterr()
    assert fp.main(["--against", str(saved)]) == 1
    out = capsys.readouterr().out
    assert f"{name}: sheet.held differs" in out and "arithmetic" in out
    with pytest.raises(SystemExit):
        fp.main(["--thorough"])  # (the thorough suite has no recorded hashes)
    with pytest.raises(SystemExit):
        fp.main(["--thorough", "--record"])
