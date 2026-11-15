# Handoff: the open issues, 15 November 2026

Where the issues stand after the pass that closed 19 of the 22 that were open
(pull requests #38 to #47), and what can be picked up next. The issues
themselves are in Beads: `bd show <id>` has the description and the
measurements of each, and `bd ready` what is open. This file only sorts them;
delete it once they are done or sorted again.

## Waiting on the owner since before

- `fusain-fdh`: calibrating graphite tones needs scans of real pencil.
- `fusain-6ru`: what is left of it makes strokes faster by changing the
  picture (tip steps of width/4, laying once per pixel, a cheaper reach curve,
  pigment stored by id, threads).
- `murex-ant` (in murex): waits on oilpaint.

## The 24 issues the pass found

All are open and none is scheduled.

### Can go now (10)

No decision is needed: the fault and the fix are both stated.

| Issue | What it is | The picture |
|---|---|---|
| `fusain-2n6` | `Sheet.rub` draws one seed per stretch, so the zeros of its pressure shift every later mark's grain | changes once: the grain is drawn again |
| `fusain-9dg` | a ballpoint drop inks all of a partly inked pixel, however small the drop | changes |
| `fusain-js5` | a light ballpoint line's blob is as wide as a firm line's | changes |
| `fusain-hay` | a flat end on a short hook cuts into the stroke before the bend (three xfail tests pin it) | changes |
| `fusain-q5k` | a film of graphite lying on dust does not shine | changes only what shines |
| `fusain-ny8` | `examples/macaw.py`: a random generator for each part of the drawing | changes once: the macaw is drawn again |
| `fusain-45t` | ink beyond the most a ball holds is not gathered: say so in the docs | unchanged |
| `fusain-238` | a short stroke is 40% Python around the kernel | unchanged, byte for byte |
| `fusain-6z7` | `examples/stilllife.py`: the cloth's folds once, and less Python around each traced stroke | unchanged, byte for byte |
| `fusain-czc` | a rub goes over each pixel's dust five times | unchanged, byte for byte; to keep only if it measures faster |

`fusain-9dg` and `fusain-js5` are both in the pen's `blot` and go together.
`fusain-ny8` comes before the macaw's feet (`fusain-9vi`): it keeps the rest
of the bird as it is when the feet change.

### Can go on a default, if the owner accepts it (4)

- `fusain-soe`, a loaded tool dirties bare paper where the hand of
  `Sheet.rub` strays past the edge of a tone. Default: a hand that strays less
  at the ends when it holds a tool, not only a note in the docs.
- `fusain-419`, a partly polished tooth glints pixel by pixel. Default: spread
  the gloss over the width of the lamp, so that no one facet catches all of
  it.
- `fusain-cyl`, shading with the side of a stick may leave stripes. The
  cross-profile can be measured and the two tests that pass narrowly pinned
  again now; whether stripes that show are believable is the owner's call.
- `fusain-2z5`, press grows with every pass, so a shading polishes in streaks.
  The fix is stated (press approaches a level set by the load), but it reworks
  `press`, as `fusain-dq2` does: do the two together.

### Need the owner's view (10)

What the right behaviour is, which the code cannot tell:

- `fusain-14f`: how well a crayon or a coloured pencil should mark a tooth
  full of pastel dust.
- `fusain-3jj`: how thick a film must be to be slick to dust.
- `fusain-bq2`: how well pastel rubbed into the paper should take a pencil.
- `fusain-6uj`: how the softness of a coloured pencil changes its burnishing;
  the rule in place was assumed, not measured.

New modelling, each at a cost:

- `fusain-r56`: a film stored apart from the dust it lies on (memory, and
  time in every rub).
- `fusain-fxc`: dust that a film has set stays set once the film is gone; it
  goes with `fusain-r56`.
- `fusain-dq2`: a second press per pixel, about 1% more memory and one more
  argument through the stroke kernels.
- `fusain-bbq`: a deep reservoir in a stump.
- `fusain-rlo`: scraped paper rougher than fresh.

The owner's eye:

- `fusain-9vi`: the macaw's feet and legs. Two critiques made without hints
  agree on the feet and found nothing wrong with the legs, which the owner
  also named: ask before calling the legs done.

## How the pass was run, for the next one

- One agent to a lane, each in its own git worktree, on issues that touch
  the same part of `_engine.py`; lanes that change the picture are merged one
  at a time, each rebased on the one before, with the fingerprint recorded
  and every image regenerated last.
- A lane that changes no picture is checked with
  `tools/fingerprint.py --thorough --save` before and `--against` after, and
  by running every example again: `git status` must show nothing.
- After any change to a kernel or a helper, `tools/refcounts.py`: a helper
  written the wrong way costs a factor of 2 to 3 (see `AGENTS.md`).
- The review bot raised three findings over the ten pull requests, and each
  was a real fault. Reproduce a finding before acting on it.
