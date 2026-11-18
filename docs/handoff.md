# Handoff: the open issues, 18 November 2026

Where the issues stand after the passes of 15 to 18 November (pull requests
#49 to #82 here, #7 to #10 in murex), and what can be picked up next. The
issues themselves are in Beads: `bd show <id>` has the description and the
measurements of each, and `bd ready` what is open. This file only sorts them;
delete it once they are done or sorted again.

## Where things stand

Two issues are open here, both waiting on the owner's hand: `fusain-fdh`
(scans of real pencil) and `fusain-586` (white wax over heavy graphite,
tried by hand, which picks how a waxy tip meets graphite crumbs). Two are
open in murex, left aside by the owner for now.

Neither library has been released since these passes. murex's requirement of
NumPy 2 and its stored spectral tables reach users only with its next
release; fusain's picture has changed several times (below) since its last.

## What the pass of 18 November did

The six issues that needed the owner's view, each decided by the owner and
closed with its measurements:

- `fusain-6uj` (#80): a coloured pencil's `melts` follows its softness
  (`Medium.pale`, `pencils.melts_at`); presets unchanged.
- `fusain-dhx`, `fusain-zhr` (#81): the press law is kept as reworked; wax
  holds as burnished from `POLISH_FLAT` (0.55) of polish on, so a
  dense dark that is not burnished comes off a vinyl eraser as before the rework
  (0.62, 0.40, 0.23 after one to three rubs). `hatching.FIRMNESS` is 0.05:
  one layer at 0.7 is uneven by 0.38 to 0.54 (was 0.52 to 0.67); what is
  left comes mostly from `STRAY_ACROSS`.
- `fusain-14f`, `fusain-3jj`, `fusain-bq2` (#82): a waxy tip rides on a bed
  of dust as a dry one does; a crayon's film over rubbed charcoal is slick
  to it; `held` counts the dust pressed into the paper apart (`ID`), which
  takes `FIXED_ROOM` of the room it took loose.

To look at, from #82: a crayon now rides on heavy graphite crumbs too (over
8B at 0.95 it lays 0.67 of what it lays on bare paper, was 0.12; a white
crayon there reads nearly clean white), which `fusain-586` is to settle;
and wax over pale pastel renders a little darker than on bare paper.

## What the passes of 16 and 17 November did

Closed, each with its measurements in the issue's closing note:

- `fusain-hn4` (#61): `firmness` is exactly nothing at the onset of
  burnishing.
- `fusain-5sj` (#62), `fusain-lir` (#66): `tools/benchmark.py` times
  processor time and runs two checkouts turn about; `--cpu-against haswell`
  compiles truly without AVX-512 (`NUMBA_CPU_NAME=haswell` alone keeps it).
  Without AVX-512 a shading takes about 0.95 of its time and a stump rub
  0.88; numba offers only process-wide settings, so fusain does not force
  it, and the README says how.
- `fusain-dpx` (#63): a stub of a shading wanders by a share of its length.
- `fusain-92i` (#64): a shading draws one number from the sheet, whatever
  the stick's wear.
- A rub's passes no longer stray a width apart and leave a band rubbed a
  fifth as much (#65; found doing `fusain-92i`, fixed rather than filed).
- `fusain-979` (#68): polish goes with the wax wherever a tool or a tip
  leaves, a dry tip's drag included.
- `fusain-ecd` (#70): `make pre-push` runs each check once and the tests on
  every core; `make tox` runs the full matrix, as CI does.
- `fusain-9vi` (#71), with the owner: the macaw's feet face the way the bird
  does (the branch slanted the wrong way in depth).
- `fusain-cyl` (#73), with the owner: a flat stick's bands meet without bare
  streaks; a shallow groove's floor bites as flattened paper.
- `fusain-6ru` (#74), with the owner: strokes and pens step a quarter of the
  tip, and the reach curve comes from a table. Shading 1.9 to 2.1 times
  faster, short strokes and pens 1.5, stump rubs 1.1 to 1.2; rubs and the
  stylus keep an eighth. Rejected, and recorded in the issue so they are not
  proposed again: a quarter for rubs (it beads), pigment by id, threads
  over tiles. Laying once per pixel per stroke (about 1.4 times more) is
  noted in the issue, not filed.
- In murex, `murex-krb` half (#9): `spectral`'s tables are stored, and
  `tests/test_bits.py` pins its numbers on x86-64.

## What the pass of 17 November evening did

The four modelling issues, with the owner:

- `fusain-bbq` (#76): a stump or a finger keeps a deep load (`Tool.deep`,
  `Smudger.deep`, `soak` once a step). One pass through a tone loads a
  stump; a used one stays grey over several drags.
- `fusain-rlo` (#77): a sand eraser or a knife raises the fibres
  (`Sheet.rough`), which take about half as much again, in blotches, until
  a firm tip, a stump or a finger lays them down.
- `fusain-r56`, `fusain-fxc` (#78): a film is stored apart from the dust
  (`Sheet.films`, `bonded` with `BOUND`); an oil pastel leaves paste and no
  film (the owner caught a finger no longer blending an oil-pastel join).
- Timed against `d110c20`: #76 and #77 together 1.01; #78 1.07 more in
  geometric mean (1.2 for a finger or kneaded eraser where every pixel has
  a dry film, 1.03 for plain graphite). Memory 0.38 to 0.6 KB a pixel.

## For the owner to look at

Results that the numbers cannot settle:

- From #76: a bare stump (cleaned before each rub) now takes 10.5% of an
  8B tone in eight rubs (5%; fusain-6lr had pinned it under 4%). A smaller
  `STUMP.deep` lowers it, and a stump stays grey for less long.
- From #77: under a light layer of HB, sanded paper reads only about 1.7 L*
  darker than fresh; `ROUGH` (1) sets how much. A ballpoint and a stylus
  ignore raised fibres, and neither the relief nor the shine shows them.
- From #78: the blender still dulls red crayon over blue pastel by a fifth
  (a dragged film goes into the tooth under the pastel: lay it back on the
  dust instead?); six layers of 6B over pastel shine 0.99 of bare paper
  against 0.82 on paper alone; films at one place are one mixture; the
  graphite of the lower film shines through a matte film over it (the review
  of #78: hiding it by the top film's matte also hides a pencil line under
  its own later samples, and moves 8 fingerprint drawings, so it needs a
  rule for which films hide). `PASTE_ROOM` (0.8) and `PLOUGH` (0.09, was
  0.12) were tuned to keep main's oil-pastel join.

Results that the numbers cannot settle:

- What `fusain-6ru` traded: tones 0.14 to 0.5 L* darker where the paper
  fills up (wax crayon 0.5 to 0.7 lighter), a slightly coarser grain, a 2%
  ripple along lines on paper without tooth, and a pencil's film over pastel
  at 0.77 to 0.80 of the line (it was 0.88 to 0.90). Five geometry tests
  have looser bounds, the measured values noted.
- What `fusain-cyl` left: a faint dark line where a flat stick's bands meet,
  1 to 5 L* with a natural hand and 3 to 9 with ruled strokes. Overlapping
  less would leave a dull line in polished wax instead.
- Test bounds that held for one sample only were widened to the spread
  measured over several sheets (#64, #65, #73, #74), the values in comments.
  A rub's rim is only asked to be rubbed at all.
- From before: a rubbed patch keeps a rim of grain not rubbed, 3 to 5 pixels
  wide (`fusain-soe`, the third row of `crayons.png`); several firm layers
  of a pencil do not add up (kept by the owner in `fusain-dhx`).

## The open issues

### Needs material from the owner (2)

- `fusain-fdh`: calibrating graphite tones needs scans of real pencil.
- `fusain-586`: the owner tries white wax over heavy graphite by hand;
  the issue lists what to do for each outcome.

### In murex (2)

- `murex-ant`: move oilpaint onto murex. It comes first: oilpaint's engine
  changes made after murex was split off are to be brought into murex, and
  oilpaint made to depend on it. This session cannot reach oilpaint.
- `murex-krb`, after `murex-ant`: on ARM eight of the bit pins differ in the
  last bits, because NumPy's SIMD `power`, `tanh` and `exp` and BLAS in the
  matrix products round by processor. The owner has decided to make
  `spectral`'s arithmetic the same everywhere (kernels of its own, no BLAS):
  a breaking change, after which fusain and oilpaint record their pictures
  again and the ARM CI job goes back in.

## How the passes were run, for the next one

- One agent to a lane, each in its own git worktree. Pull requests go
  through one branch, one at a time; a lane that changes the picture
  records its fingerprints again and regenerates every image on top of what
  merged before it.
- The squash merges leave a lane's base commit off main: bring a lane
  across as the difference between its head and its base, not with a merge
  (`git diff --binary <base> <lane> | git apply --index`, or
  `git read-tree -u --reset <lane>` when its base has the tree of main).
- What suited which lane: the strongest model for kernels and anything that
  measures timings; the next for modelling in `sheet.py` and the render; a
  middle one for tools and Python with a stated rule; the smallest for a
  mechanical change, told to stop if the fault does not reproduce.
- An issue that needs the owner's eye or choice (`fusain-9vi`,
  `fusain-cyl`, `fusain-6ru`) goes to a session of its own with the owner:
  it measures first, proposes, and waits for the owner before changing the
  picture. It does not touch this file; the parent session does.
- A fault found on the way is fixed, in its own lane or pull request,
  rather than filed. A test that holds for one seed is made to hold over
  several.
- Timings run one at a time on an idle machine: processor time, both
  versions as live processes taking each workload turn about, best of many
  (`tools/benchmark.py --against`), and on an AVX-512 machine again with
  `--cpu-against haswell`.
- A lane that changes no picture is checked with
  `tools/fingerprint.py --thorough --save` before and `--against` after,
  and by running every example again: `git status` must show nothing.
  After any change to a kernel or a helper, `tools/refcounts.py`.
- A git worktree under a checkout that has `node_modules` or a warm cache
  hides what a fresh clone shows: reproduce such a fault in a clone
  outside it.
- A stale `.beads/issues.jsonl` in a lane can reopen issues closed since it
  branched (caught in #74's review): import main's before committing.
- Lanes, each in a git worktree, share the main checkout's Beads database, and bd's
  commit hook exports it, other lanes' closes included, into the main
  checkout: commit only a lane's own issue lines, on main's file.
- Look at a change to the picture as a delta (before, after and a heatmap
  of ΔE), not only as two renders: the owner found the oil-pastel fault so.
- The review bot's findings have been real when it had any (the still life
  under NumPy 1.x; the compatibility claim of murex #9). Reproduce a
  finding before acting on it.
