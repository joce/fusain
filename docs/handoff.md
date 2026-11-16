# Handoff: the open issues, 16 November 2026

Where the issues stand after the pass of 15 and 16 November (pull requests
#49 to #60 here, #7 and #8 in murex), and what can be picked up next. The
issues themselves are in Beads: `bd show <id>` has the description and the
measurements of each, and `bd ready` what is open. This file only sorts them;
delete it once they are done or sorted again.

## What the pass did

Sixteen issues closed, twenty open.

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

## For the owner to look at

Three results of the pass that the numbers cannot settle:

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
- Neither library has been released since. murex's requirement of NumPy 2
  reaches users only with its next release.

## Waiting on the owner since before

- `fusain-fdh`: calibrating graphite tones needs scans of real pencil.
- `fusain-6ru`: what is left of it makes strokes faster by changing the
  picture.
- `fusain-cyl`: shading with the side of a stick may leave stripes; left
  open on 15 November, not to be measured or pinned again for now.
- `murex-ant` (in murex): waits on oilpaint.

## The twenty open issues

### Can go now (5)

No decision is needed: the fault and the fix are both stated.

| Issue | What it is | The picture |
|---|---|---|
| `fusain-92i` | `Sheet.shade` with a `Stick` draws a number of strokes that follows the stick's wear, so every later mark's grain follows the pressure of what was drawn before | changes once: the grain after such a shading |
| `fusain-dpx` | a stub of a shading at a corner of its area wanders as far as a long stroke, and bends through a right angle | changes: the stubs |
| `fusain-hn4` | `firmness(0.4)` is not exactly 0 and leaves a vanishing number in the press arrays | by a bit at most |
| `fusain-5sj` | `tools/benchmark.py` times wall clock and has no stroke with flat ends | unchanged |
| `fusain-lir` | kernel loops over single-precision arrays are compiled 512 bits wide on AVX-512 machines, which costs a rub a tenth: measure strokes too | unchanged; the fix, if any, must be an option numba documents |

`fusain-5sj` comes before `fusain-lir`: it gives the benchmark the
processor time and the two code generations that `fusain-lir` needs.

### Can go on a default, if the owner accepts it (1, and 1 in murex)

- `fusain-979`, polish exceeds what the wax holds by up to 0.004 until the
  next lift, and a melting tip can polish a pixel where it lays no wax.
  Default: call `bare_tooth` after `push_along` in `rub_under`, and let a
  tip raise polish only where the pixel holds wax when it leaves.
- `murex-krb` (in murex), `spectral`'s numbers follow NumPy's own
  arithmetic to the last bit. Default: store `_BASIS` and
  `SPECTRUM_TO_RGB` as constants, and pin a few reflectances to the bit in
  a test. It is a change to `spectral`, so it needs oilpaint checked.

### Need the owner's view (14)

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

The owner's eye:

- `fusain-9vi`: the macaw's feet and legs. Since `fusain-ny8` a change to
  the feet stays in the feet (2.6% of the drawn pixels move, where 82%
  did), so they can be redrawn without the rest of the bird changing. Ask
  before calling the legs done.

## How the pass was run, for the next one

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
  many; and on an AVX-512 machine again with `NUMBA_CPU_NAME=haswell`
  (`fusain-lir` nearly gave `fusain-czc` a false gain).
- A git worktree under a checkout that has `node_modules` or a warm cache
  hides what a fresh clone shows: reproduce such a fault in a clone
  outside it.
- The review bot raised one finding over twelve pull requests, and it was
  a real fault (the still life under NumPy 1.x). Reproduce a finding
  before acting on it.
