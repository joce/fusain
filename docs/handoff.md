# Handoff: the open issues, 16 November 2026

Where the issues stand after the two passes of 15 and 16 November (pull
requests #49 to #71 here, #7 to #9 in murex), and what can be picked up
next. The issues themselves are in Beads: `bd show <id>` has the
description and the measurements of each, and `bd ready` what is open. This
file only sorts them; delete it once they are done or sorted again.

## What the passes did

The first pass (#49 to #60): sixteen issues closed, twenty open.

- The ten that could go: `fusain-2n6`, `fusain-9dg`, `fusain-js5`,
  `fusain-45t`, `fusain-q5k`, `fusain-238`, `fusain-6z7`, `fusain-hay`,
  `fusain-ny8`, `fusain-czc`. The last was an experiment and its answer is
  no: a rub's five passes over a pixel's dust stay as they are.
- Three of the four that waited on a default, which the owner gave:
  `fusain-soe`, and `fusain-2z5` with `fusain-dq2` (the rework of press:
  `Sheet.crush` and `Sheet.polish`). The fourth, `fusain-cyl`, the owner
  left open.
- `fusain-419` was cleared by the rework of press, not by the lamp with a
  width that was built for it; the owner dropped the lamp. Its rule and
  numbers are in the issue's closing note.
- Found and closed on the way: `fusain-klv` (NumPy 1.x is dropped, here and
  in murex: `numpy>=2.0`, `numba>=0.60`) and `fusain-t6h` (`make pre-push`
  in a fresh clone).
- New in the library: a `seed=` on `Sheet.stroke`, `shade`, `smudge`,
  `erase` and `rub`.

The second pass (#61 to #71): seven closed, thirteen open, none of them
able to go without the owner. `fusain-9vi`, the macaw's feet, was done with
the owner in a session of its own (#71); `make pre-push` now runs each
check once, and `make tox` the full matrix (#70).

The five that could go are closed: `fusain-hn4` (#61), `fusain-5sj` (#62),
`fusain-dpx` (#63), `fusain-92i` (#64) and `fusain-lir` (#66). A fault found
on the way was fixed rather than filed (#65): a rub's passes could stray a
width apart and leave a band rubbed a fifth as much as the middle.
`fusain-lir` found that `NUMBA_CPU_NAME=haswell` alone keeps AVX-512: the
benchmark's `--cpu` now empties `NUMBA_CPU_FEATURES`, and the README says
how to compile without AVX-512 (shading about 0.95 of the time, a stump rub
0.88; fusain does not force it).

## For the owner to look at

Results of the passes that the numbers cannot settle:

- A rubbed patch keeps a rim of grain that is not rubbed, 3 to 5 pixels
  wide along its sides, now that a tool stays inside its area
  (`fusain-soe`, pull request #56): the third row of `crayons.png`. The
  alternative, a tool that straddles the edge, leaves dust outside that
  shows.
- A dense dark that is not burnished comes off a vinyl eraser nearly to its
  stain in three rubs (0.05 left; it was 0.27), and several firm layers of
  a pencil no longer add up (three of HB flatten 0.34 of the tooth; it was
  0.74). Both are in `fusain-dhx`, with the constants that would change
  them.
- New grain moved test bounds that held for one sample only. In #64 nine
  were widened to the spread measured over several sheets, the values in
  comments. In #65 a rub's rim is only asked to be rubbed at all (0.04 of
  the middle failed in 17% of samples, before the fix as after), and an
  eraser's rows from the rim in to 0.5 of the middle: its ±15% of
  firmness leaves it short even with a steady hand.
- Neither library has been released since. murex's requirement of NumPy 2
  reaches users only with its next release.

## Waiting on the owner since before

- `fusain-fdh`: calibrating graphite tones needs scans of real pencil.
- `fusain-6ru`: what is left of it makes strokes faster by changing the
  picture.
- `fusain-cyl`: shading with the side of a stick may leave stripes; left
  open on 15 November, not to be measured or pinned again for now.
- `murex-ant` (in murex): waits on oilpaint.

## The thirteen open issues, and `murex-krb`

### Done on the owner's defaults (#68 here, #9 in murex)

- `fusain-979` is closed: polish goes with the wax wherever a tool or a tip
  leaves, a dry tip's drag included (#68).
- `murex-krb` is half done and stays open: `spectral`'s tables are stored,
  and `tests/test_bits.py` pins its numbers on x86-64 (murex #9). On ARM
  eight of those pins differ in the last bits. The owner has decided to make
  spectral's arithmetic the same everywhere (a breaking change: fusain and
  oilpaint recorded again), but only after oilpaint's later engine changes
  are brought into murex and oilpaint depends on murex.

### Need the owner's view (13)

What the right behaviour is, which the code cannot tell:

- `fusain-dhx`: the choices of the rework of press (what several layers
  do, how fast wax that is not burnished comes off an eraser, its
  constants, 4 to 8% on strokes).
- `fusain-zhr`: the hand's ±15% of firmness makes neighbouring strokes
  differ four-fold in polish near the onset of burnishing. It is what
  still polishes a shading in streaks.
- `fusain-14f`: how well a crayon or a coloured pencil should mark a tooth
  full of pastel dust.
- `fusain-3jj`: how thick a film must be to be slick to dust.
- `fusain-bq2`: how well pastel rubbed into the paper should take a pencil.
- `fusain-6uj`: how the softness of a coloured pencil changes its
  burnishing.

New modelling, each at a cost:

- `fusain-r56`: a film stored apart from the dust it lies on. It would also
  take out what `fusain-q5k` had to assume about a film of graphite.
- `fusain-fxc`: dust that a film has set stays set once the film is gone;
  it goes with `fusain-r56`.
- `fusain-bbq`: a deep reservoir in a stump.
- `fusain-rlo`: scraped paper rougher than fresh.

Waiting since before, as said above: `fusain-6ru`, `fusain-fdh`,
`fusain-cyl`.

## How the passes were run, for the next one

- One agent to a lane, each in its own git worktree. Pull requests go
  through one branch, one at a time: a lane is rebased on main with
  `git rebase --onto main <its base>`, its fingerprints recorded again,
  every image regenerated, and what `git status` shows compared with what
  the lane said would change.
- What suited which lane: the strongest model for kernels (`fusain-hay`,
  `fusain-czc`, the rework of press); the next for modelling in `sheet.py`
  and the render; a middle one for examples and for Python with a stated
  rule; the smallest for a mechanical change, told to stop if the fault
  does not reproduce (it did stop, rightly).
- Order lanes that may share a cause. `fusain-419` was built against a
  symptom that `fusain-2z5` removed: the lamp was measured, and dropped.
- A lane that changes no picture is checked with
  `tools/fingerprint.py --thorough --save` before and `--against` after,
  and by running every example again: `git status` must show nothing.
- After any change to a kernel or a helper, `tools/refcounts.py`.
- Timings on a machine that does anything else: processor time, the two
  versions as live processes taking each workload turn about, best of
  many (`tools/benchmark.py --against`); and on an AVX-512 machine again
  compiled without it, `--cpu-against haswell` (`NUMBA_CPU_NAME=haswell`
  alone keeps the host's AVX-512 features).
- A git worktree under a checkout that has `node_modules` or a warm cache
  hides what a fresh clone shows: reproduce such a fault in a clone
  outside it.
- The squash merges leave a lane's base commit off main: bring a lane
  across as the difference between its head and its base, not with a merge
  (`git diff --binary <base> <lane> | git apply --index`, or
  `git read-tree -u --reset <lane>` when its base has the tree of main).
- A lane that finds a fault fixes it, as a lane of its own after the one
  that found it, rather than filing it; a test that holds for one seed is
  made to hold over several.
- The review bot raised one finding over eighteen pull requests, and it was
  a real fault (the still life under NumPy 1.x). Reproduce a finding
  before acting on it.
