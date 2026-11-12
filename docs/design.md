# Core model: design proposal

Status: accepted (issue `fusain-ilv`). Steps 1 and 2 of the build order are built; the rest is not.

## What fusain has to get right

A dry medium is a solid stick that the paper wears down. The mark is whatever
the paper scraped off and kept. Everything characteristic about pencil,
charcoal and pastel follows from that, and these are the behaviours the model
is judged on. Each one is measurable, so each becomes a test.

1. **Grain.** At light pressure only the peaks of the paper's tooth take
   pigment; the valleys stay white. Pressing harder reaches further down, and
   the valleys fill last.
2. **Saturation.** Going over the same place darkens it less each time and
   then stops: the tooth is full. A hard lead stops at a lighter tone than a
   soft one. Rough paper takes more layers than smooth paper.
3. **Loose and held pigment.** Charcoal and soft pastel leave dust that a
   finger moves and a kneaded eraser lifts. Graphite is held better, wax-based
   pencil and crayon almost completely.
4. **Layering.** Coloured pencil is translucent: yellow over blue reads green.
   Pastel is opaque: a light stick covers a dark one.
5. **Wax.** A waxy layer fills the tooth and turns slick: later layers take
   badly (resist), and heavy pressure polishes it flat (burnishing).
6. **The tip.** A point wears to a flat, so a line widens as it is drawn.
   The side of a stick lays a broad band. Tilting the pencil widens the mark.
7. **Conservation.** What leaves the stick is on the paper, on the tool that
   moved it (stump, eraser), or counted as dust that fell off.

## The state: a sheet of paper

Per pixel, all `float32`:

| Field | What it holds |
| --- | --- |
| `tooth` | Height of the paper surface, 0 (valley) to 1 (peak). Set once from the paper's texture. |
| `press` | How far the tooth has been crushed or polished here (0 to 1): burnishing, and lines indented with a stylus. |
| `cover` | Share of the pixel that carries pigment (0 to 1); the rest shows bare paper. |
| `held` | Pigment fixed to the paper: absorption K over 24 spectral bands, scattering S, volume, and how much of the volume is wax. |
| `loose` | Pigment lying on the surface as dust: K, S and volume. |

That is about 0.23 KB per pixel (oilpaint needs 0.35). As there, storing
summed absorption and scattering lets any number of pigments mix correctly at
a fixed cost.

Two layers, `held` and `loose`, play the part of oilpaint's two wet layers.
The difference between them is what can still move: a stump, an eraser or a
later stroke shifts `loose`; only a hard eraser or a blade touches `held`.
Pressure moves pigment from `loose` to `held`. So does fixative.

Sizes are in pixels, and the paper's grain sets the scale of a drawing: the
tooth of a drawing paper is finer than a pencil line, which is two or three
grains wide. (The first version of the tooth had patches several grains
across, and shading came out in blotches wider than the pencil.)

The paper itself is described by a `Paper` (texture, tooth size in pixels,
depth of the tooth, how abrasive it is, how much it gives under a tip, colour), with presets such as
cartridge paper, hot-pressed, cold-pressed, laid (Ingres) and sanded pastel
paper.

## A medium

A medium is a pigment plus a handful of physical numbers. The same stroke
code runs all of them; only the numbers change.

| Parameter | Meaning | Low | High |
| --- | --- | --- | --- |
| `softness` | How easily it wears, and how far it gives to reach into the tooth | 6H pencil, hard pastel | 8B pencil, vine charcoal, soft pastel |
| `binding` | Share of the deposit that is held at once rather than left as dust | vine charcoal, soft pastel | coloured pencil, wax crayon |
| `wax` | Binder that films over the tooth: resist, burnishing, gloss | graphite, charcoal | crayon, oil pastel |
| `opacity` | How strongly it hides what is underneath (as in oilpaint's `Paint`) | coloured pencil | pastel |
| `sheen` | Metallic lustre of the laid pigment | everything else | graphite |

Presets group by family: `pencils.HB`, `pencils.B6`, `charcoal.VINE`,
`charcoal.COMPRESSED`, `pastels.soft("#c0392b")`, `crayons.wax(...)`, and so
on. Pigment colour works as in oilpaint: a colour is turned into K and S
spectra.

## A tool

A tool is a tip plus, usually, a medium.

- **Tip.** A shape (point, flat, the side of a stick) and a size. In the first
  version the tip is a formula: a point whose flat grows with the volume worn
  off, and `sharpen()` to reset it. A worn tip stored as its own small height
  map (so a crayon keeps the facets it wore) can come later.
- **Tools that lay nothing** use the same stroke runner: stump and finger
  (carry `loose` along and press part of it in), kneaded eraser (lifts
  `loose`), hard eraser (also abrades `held`), burnisher.
- **Fixative** acts on the whole sheet: `loose` becomes `held`.

## One step of a stroke

A stroke is sampled along its path as in oilpaint; pressure can vary along it
(tilt and speed are still to come). At each step, for each pixel under the
tip:

1. **Reach.** Pressure, the softness of the medium and the give of the paper
   set how far down the tooth the tip gets. A tip is flat across with a
   rounded rim. A light touch reaches further than in proportion, and only the
   softest leads pressed hard reach every valley: a firm HB line still has
   white specks in it. No two passes meet the paper alike: each pass shifts
   the level it reaches a little, at random, pixel by pixel. That is what lets
   light layers build tone.
2. **Contact.** A pixel is not level, so a tip that reaches near its height
   touches part of it. That share becomes the pixel's `cover`, which only
   grows: the highest parts are always the first touched. The tip bears
   hardest on what stands highest, so the high ground takes more than the
   low: that is the grain inside a line. A lead is not even either: across
   the tip it bears more in some bands than in others, which leaves fine
   streaks along the stroke.
3. **Room.** The tooth holds only so much: more if it is deep and abrasive,
   more of a soft lead than of a hard one, more under a firm touch than a
   light one. What a pass lays shrinks as the room runs out.
4. **Deposit.** What one pass lays depends on the medium, the pressure and
   the paper's bite, not on how much the tooth holds; so a toothier paper
   takes more passes to fill. It is split between `held` and `loose` by the
   medium's `binding`, and it is what the tip loses.
5. **Working what is there** (still to come). The tip presses some `loose`
   into `held` and drags some of it along. High pressure on a waxy layer
   raises `press`.

These are numba kernels under the same rules as oilpaint's (no `fastmath`, no
`parallel`), so a drawing repeats exactly. Their constants are in
`fusain/_engine.py`. They were set so that tones land near these figures,
which are estimates from experience, not measurements of real drawings: on
cartridge paper, one even layer of HB reflects about 0.8 of what the paper
does at a light pressure and 0.4 at full pressure; 6B about 0.5 and 0.13; 4H
at full pressure about 0.75; and light layers of HB level off near 0.5.

## Rendering

Per pixel: Kubelka-Munk reflectance of the `loose` layer over the `held` layer
over the paper's own colour, on the covered share of the pixel; the rest shows
bare paper.
By default a fairly high lamp shows the tooth faintly, as on a scan; a raking
light brings it out, and `lighting=False` renders the sheet flat. Still to
come: a specular term for graphite's shine where it is thick and polished.

## What the prototype showed

A throwaway numpy script ran an early form of this step law with a grey ramp
for colour, before anything was built:
`docs/design/prototype.png`. Rows, top to bottom: HB hatching with pressure
rising left to right; 6B the same; the side of a charcoal stick with a stump
dragged through it; one, two, four and eight light layers of HB.

- **Works:** grain at light pressure, valleys filling as pressure rises, a
  soft lead far darker than a hard one at the same pressure (mean reflectance
  0.28 against 0.83 near full pressure), a broad grainy charcoal band.
- **Did not work:** layering. Eight light passes are barely darker than
  one (0.897 against 0.913), where a real pencil builds tone steadily. The
  contact rule was too exact: every pass touched the same pixels. The step
  law above now has a random part per pass, and eight light layers of HB come
  out at 0.62 against 0.85 for one.
- **Too crude:** the stump blurred to a clean grey band. It should leave
  streaks along its direction (step 3).

## What is shared with oilpaint

The shared code lives in a third package, `murex`: the colour science
(colour to spectrum, Kubelka-Munk, spectrum to sRGB), pigments, paths and
profiles along a stroke, noise (which already has a paper texture), PNG
output and reference images. Not shared: oilpaint's paint engine, brushes and
knives, which model a liquid.

## Build order

1. **Paper and sheet.** `Paper`, `Sheet`, rendering of blank paper.
2. **Graphite.** The step law as a kernel, pencil grades, pressure. Tests for
   behaviours 1, 2 and 7, with measured tone against pressure and against
   number of passes.
3. **Dust.** `loose` and `held`, stump, erasers, charcoal. Behaviour 3.
4. **Colour.** Coloured pencil and pastel: layering, wax, burnishing.
   Behaviours 4 and 5.
5. **Tips.** Wear, tilt, the side of a stick, crayon. Behaviour 6. Graphite
   shine in the renderer.
6. **Extras.** Fixative, hatching helpers, ballpoint pen.

Ballpoint is last because it is a different mechanism: a paste the ball rolls
on at a steady rate, not something the paper scrapes off. It fits the same
sheet (fully held, transparent, indents the paper) with its own deposit rule.

## Sketch of the API

```python
from fusain import Sheet, papers, pencils, pastels, tools

sheet = Sheet(1200, 800, paper=papers.CARTRIDGE, seed=1)
sheet.stroke([(100, 300), (300, 220), (500, 320)], pencils.HB, pressure=0.4)
sheet.stroke(path, pastels.soft("#c0392b"), pressure=(0.2, 0.8), side=True)
sheet.smudge(path, tools.STUMP)
sheet.erase(path, tools.KNEADED)
sheet.fix()
sheet.save("drawing.png")
```

## Decisions to make

1. **Sharing code with oilpaint.** Decided: a third package, `murex`, holds
   the shared modules. fusain uses it from the start; oilpaint moves onto it
   later.
2. **Units.** Pixels, as in oilpaint (proposed), or millimetres with a
   resolution, so that a 0.5 mm lead and an A4 sheet mean something.
3. **Names.** `Sheet` and `Paper` (proposed) or oilpaint's `Canvas` and
   `Surface`.
4. **Where to start.** Graphite first (proposed: one pigment, the purest test
   of the deposit rule), or colour from the start.
