# Core model: design proposal

Status: accepted (issue `fusain-ilv`). All six steps of the build order are built.

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
   pencil and crayon almost completely, and wax holds against an eraser: a
   light layer of coloured pencil comes off a vinyl one, a dense or
   burnished one hardly, and its lights are kept or scratched out. An oil
   pastel is a paste: a finger smears it, and a kneaded eraser does not lift
   it.
4. **Layering.** Coloured pencil is translucent: yellow over blue reads green.
   Pastel is opaque: a light stick covers a dark one.
5. **Wax.** A waxy layer fills the tooth and turns slick: later layers take
   badly (resist), dry media hardly at all, even over one firm line (charcoal
   shaded over a crayon drawing leaves it showing), and heavy pressure
   polishes it flat (burnishing): a colourless blender pressed hard closes
   the white flecks of the paper, and the colour is deeper, smoother and
   glossy. Laid over dust, wax lies on it and binds part of it.
6. **The tip.** A point wears to a flat, so a line widens as it is drawn.
   The side of a stick lays a broad band. Tilting the pencil widens the mark.
7. **Conservation.** What leaves the stick is on the paper, on the tool that
   moved it (stump, eraser), or counted as dust that fell off.
8. **The surface.** The paper is worked as well as marked. A hard tip pressed
   firmly flattens the tooth, which is then touched all over and bites less.
   A stylus, or the ball of a pen, indents a groove, and a tone shaded over
   it afterwards skips it: the line stays the colour of the paper. And what
   lies flat shines under a lamp: graphite like metal, polished wax softly,
   heavy ballpoint ink like bronze.

## The state: a sheet of paper

Per pixel, all `float32`:

| Field | What it holds |
| --- | --- |
| `tooth` | Height of the paper surface, 0 (valley) to 1 (peak). Set once from the paper's texture. |
| `press` | How far the tooth has been flattened or polished here (0 to 1): by a hard tip pressed firmly, and by burnishing wax. A tooth pressed flat is level: a tip touches all of it, as it would its peaks, and in the relief its peaks have come down to its valleys. |
| `dent` | How far the paper has been indented here: the depth of a groove below the peaks of the tooth, in depths of the tooth of a paper of depth 1. By a stylus, and by the ball of a pen. |
| `cover` | Share of the pixel that carries pigment (0 to 1); the rest shows bare paper. |
| `held` | Pigment fixed to the paper: absorption K over 24 spectral bands, scattering S, volume, how much of the volume shines (graphite) and how much is wax. The wax counts all the wax at the pixel, wherever it lies: in what the paper holds, in a film or in paste. |
| `ink`, `inked` | The ink of pens, a stain in the paper under everything else: K, S and volume, and the colour it bronzes (linear RGB, times the volume); and the share of the pixel it is on. Allocated when a pen first draws. |
| `under`, `loose` | Pigment lying on what the paper holds: K, S, volume and sheen, in two layers. `loose` is on top; what a new pass covers goes `under`. Most of it is dust. |
| `bonded` | Of what lies in each of the two layers, the volume that is not loose dust, of three kinds: `fixed`, dust that is set, by fixative or under a film; `film`, what a waxy medium laid on the dust and not on the paper; `paste`, what the paper does not hold of a greasy medium. |

That is about 0.37 KB per pixel, a little more than oilpaint. As there, storing summed
absorption and scattering lets any number of pigments mix correctly at a fixed
cost.

`held` is what is fixed; the two layers of dust are what can still move: a
stump, an eraser or a later stroke shifts them, and only a hard eraser or a
blade touches `held`. Pressure moves pigment from the dust to `held`.
Graphite is the exception: it slides. Under a stump, a finger or a chamois
the graphite that the paper holds is pushed from grain to grain, and a little
of it comes loose, the more the softer the lead (the clay of a hard lead
holds it back); so a soft pencil's tone can be rubbed smooth and dragged out,
and a hard one's hardly.
Fixative sets the dust where it lies. There are two layers of dust because order matters: dust of two
pigments in one layer is a mixture, and in a mixture of light and dark the
dark wins, whereas a light pastel laid over a dark one covers it. So a pass
lays its dust on top, and the dust that was there goes under it; rubbing
stirs the two together.

Not all that lies in the two layers is dust, and `bonded` says how much of
each is not. The rule is that a medium lies on what it meets:

- **A film.** What the paper holds of a waxy medium (a coloured pencil, a
  crayon, an oil pastel) is a film of wax, and a film sticks to whatever is
  under the tip. On bare paper, or over what the paper holds already, that
  is `held`, as for any medium. Where dust lies on the paper, the film lies
  on the dust: it goes on top with the dust of the pass, as far as the dust
  covers the paper (all of it over dust as thick as covers what is under
  it, less over thinner dust, through which the tip reaches the paper). It
  is seen over the dust, a later pass buries it as it buries dust, and it
  is as fast as what the paper holds: only a hard eraser takes it off. A dry
  medium has no film: what the paper holds of a pencil laid over charcoal
  dust is in the tooth, under the dust.
- **Dust that is set.** A film binds part of the loose dust it is laid
  over: seven tenths of it under a film thick enough to cover it, less under
  a thinner one. That dust is set where it lies, exactly as fixative sets
  it, and counts with it in `fixed`.
- **Paste.** What the paper does not hold of a greasy medium (a waxy one
  that is soft: an oil pastel) is not dust but paste. It lies in the layers
  as dust does, so a light oil pastel shows over a dark one, and a tip
  presses it in as it does dust; but it sticks. A tool moves it only as far
  as it smears: a finger, a stump or a chamois drags it about and blends it,
  and it is paste again where they lay it; an eraser, which lifts and does
  not smear, takes it only as it takes what the paper holds. Dust under a
  pass of paste turns to paste in the same proportion as dust under a film
  is set, and a greasy tip ploughs up a little of what it is laid over, so
  that a light oil pastel only half covers a dark one. Fixative sets paste.

The surface has two fields because a tip meets it in two ways. Where a
whole area has been pressed flat, the tip that comes next rides on it: what
matters is that it is level, so to a tip `press` raises the low ground to
the peaks, and the tip touches all of it. A groove is often narrower than
the tip, which then rides on the paper either side of it: what matters
there is how far below that the floor is, so `dent` is a depth, and
nothing in a groove stands higher than `1 - dent / depth` (the peaks of
this paper's tooth being at 1). A light dent only takes the peaks off; a
deep one is a level floor below the valleys. A tip rests on the highest of
the paper it covers, so one narrower than the groove under it goes down
into it, as far as all it covers is indented; the sheet keeps no width of
a groove, since the dent around the tip says it, for grooves of any width,
crossed or pressed over each other. Where the point that pressed it took
the peaks off, the floor is flat, as a tooth pressed flat is, and bites a
dry medium as little. Both show in the
relief the lamp lights. There a flattened patch has been pressed down: its
peaks come down towards its valleys, by `press` of their height, so that
pressed flat it is level with them, a little below the paper around it,
and a valley the tip never reached is not left a pit. (They were raised to
the peaks, as for a tip, and such a valley was left a pit as deep as the
tooth: six firm layers of 4H left a twentieth of the patch more than a
quarter of a tooth below the rest, and under a raking light the patch
looked as rough as the paper left alone; now none, and a seventh as rough.)
A groove is a line of shadow under a raking light.

Sizes are in pixels, and the paper's grain sets the scale of a drawing: the
tooth of a drawing paper is finer than a pencil line, which is two or three
grains wide. (The first version of the tooth had patches several grains
across, and shading came out in blotches wider than the pencil.)

The paper itself is described by a `Paper` (texture, tooth size in pixels,
depth of the tooth, how abrasive it is, how much it gives under a tip, colour), with presets such as
cartridge paper, hot-pressed, cold-pressed, laid (Ingres), pebbled (charcoal
paper) and sanded pastel paper.

A texture is judged by the scales at which its tooth varies, since a tone laid
on it varies at the same ones. A wove paper varies at a grain and below: a
tone on it is fine flecks. A rough paper also varies over many grains, so at
a small grain a tone on it is blotchy. A pebbled paper is pits a grain
across, 1.3 grains apart, each with its own depth and outline, in a net of
rims: it varies at the scale of a pit and little beside (at grain 4, a tenth
of its variation is at wavelengths over four grains, against over half for a rough
paper), and a light tone on it is a net of dark around pale pits.

## A medium

A medium is a pigment plus a handful of physical numbers. The same stroke
code runs all of them; only the numbers change.

| Parameter | Meaning | Low | High |
| --- | --- | --- | --- |
| `softness` | How easily it wears, and how far it gives to reach into the tooth | 6H pencil, hard pastel | 8B pencil, vine charcoal, soft pastel |
| `binding` | Share of the deposit that is held at once rather than left as dust | vine charcoal, soft pastel | coloured pencil, wax crayon |
| `wax` | Binder that films over the tooth: resist, burnishing | graphite, charcoal | coloured pencil, crayon, oil pastel |
| opacity (of the pigment) | How strongly it hides what is underneath | coloured pencil | pastel |
| `sheen` | How much of it mirrors the light once the paper holds it, as a metal does | everything else | graphite, the more the softer the lead |
| `melts` | How far it melts into a film of wax it is pressed hard on, and stirs it | everything else | a colourless blender |

Presets group by family: `pencils.HB`, `pencils.B6`, `charcoal.VINE`,
`charcoal.COMPRESSED`, `pastels.soft("#c0392b")`, `pastels.oil(...)`,
`crayons.wax(...)`, and so on. A wax crayon is all held, translucent and
waxier than a coloured pencil, so it mixes with what is under it and does not
cover it. An oil pastel is a greasy paste: less than half of it is held at
once, and the rest lies in the two layers as paste, where a finger smears
it, a later layer half covers it and a kneaded eraser does not lift it.
Pigment colour works as in oilpaint: a colour is turned into K and S
spectra.

## A tool

A tool is a tip plus, usually, a medium.

- **Tilt** (built). A stick held on its side marks with the side of its lead:
  a band across the stroke, as wide as the length of the side that lies
  on the paper, whose weight is spread over more paper, so the mark is
  lighter and grainier. How long the side is depends on
  the stick: the lead a pencil's sharpening bared is about five times its
  point, a piece of vine charcoal laid flat eight times its end. A medium
  says it as a multiple of the point (`Medium.side`), and a `Stick` can give
  it in pixels (`Stick(charcoal.VINE, width=6, lead=6, side=90)`), for a
  band as wide as the stick is long. Part way over, the mark is the point
  and that share of what the side adds. What the side is decides how it
  bears (`Medium.flat_side`). The side of a pencil is the cone of lead its
  sharpening bared, which bears hardest along its middle: an ellipse across
  the stroke, its band darkest down the middle and soft at the edges. A
  stick of charcoal, pastel or crayon laid flat bears alike all along its
  length: its band is even across and ends where the stick does, rounded
  within the stick's thickness. The side of a stick of vine charcoal 90
  pixels long and 6 thick, on a level paper, lays 0.92 of its band at 0.9
  of its middle or more, and falls from there to a tenth in 3.5 pixels;
  as an ellipse it laid 0.43, and fell over 23.5 (on cartridge paper 0.70,
  from 0.28 to 0.31, for charcoal and pastel alike). Where the band is
  put down and lifted it is cut straight across, as the side of a stick
  lands and lifts all along its length at once, and a shading lays such
  bands side by side, overlapping only by their rounded ends. (It was an
  ellipse for every medium.)
- **Wear** (built). A `Stick` is a medium in the hand, with the state of its
  tip. A point is a cone, which loses volume as the cube of where it is cut:
  the cube of the flat's width grows with the volume the tip has laid, up to
  the full width of the lead, so a line widens as it is drawn and a soft lead
  blunts soonest. `sharpen()` gives the point back. A bare medium draws with a
  tip that keeps its width.
- **Still to come.** A worn tip stored as its own small height map, so that a
  crayon keeps the facets it wore.
- **A stylus** (built): `Sheet.indent(path, pressure, width)`. A hard round
  point that lays nothing: it presses a groove, as deep under its middle as
  it is pressed hard and as the paper gives (two depths of a tooth at full
  pressure on a paper of middling give), shallower towards its rim; the
  deepest pass is what stays. A pencil, a crayon or charcoal shaded over it
  rides on the paper either side and marks only what it reaches down to,
  and the floor, pressed flat, bites it less: on cartridge paper a 2B at
  any pressure lays nothing in a groove indented at 0.7, and a 6B pressed
  hard a fiftieth of what it lays beside it; in one indented at 0.3 a 2B
  shaded lightly lays an eighth and a 6B pressed hard two thirds; a hard
  lead reaches less far down than a soft one. That
  is how whiskers and fine highlights are kept in a pencil drawing. A tip
  narrower than the groove follows it down: an HB point a pixel wide drawn
  along a groove indented at 0.7 by a stylus 3 wide lays on its floor 0.75
  of what it lays on the paper left alone, and touches all of it; a 2B
  point 3 wide 0.49, and one 4 wide, which rides on the paper either side,
  nothing. (Every tip rode over a groove, and laid nothing in that one.)
  What was drawn there before goes down with the paper, and no eraser takes
  a groove out.
- **Tools that lay nothing** use the same stroke runner: stump and finger
  (carry `loose` along and press part of it in, and slide held graphite),
  kneaded eraser (lifts loose dust), hard eraser (also abrades `held`, and
  with it what is set, a film and paste), and what scrapes: a sand eraser
  and the point of a knife.
- **Wax holds against an eraser** (built). Wax binds what it holds, and an
  eraser slides on it. Of what the paper holds at a pixel (and of a film
  and what is set there) an eraser lifts `WAX_HOLDS` less where the wax is
  `WAX_FAST` thick or more and has kept its tooth, less in proportion where
  it is thinner, and `POLISH_HOLDS` less where it has been polished flat
  (`press`). One rub to and fro of a vinyl eraser at 0.9 takes nine tenths
  of one light layer of a dark coloured pencil, two thirds of three layers
  at 0.5, and a third of four firm layers, burnished or not; three rubs
  leave three tenths of the firm layers, lighter by then (a fifth of the
  paper's tone), and two fifths of the burnished ones, as dark as they
  were. A kneaded eraser takes a quarter of the light layer in a rub and a
  fourteenth of the burnished ones. (Before, a vinyl eraser took nine
  tenths of anything in a rub, and a kneaded one a third.) A tool with
  `scrape` cuts through the wax by that share: a sand eraser
  (`tools.SAND_ERASER`, 0.8) takes nine tenths of the burnished dark in a
  rub, and the point of a knife (`tools.KNIFE`, 1) scratches a line two
  pixels wide back to the paper (it takes all but a hundredth of what lay
  on its line, and the wax with it, so the line takes pencil again). What
  they take off is on the tool, as for any eraser. The lights of a coloured
  pencil drawing are therefore kept (bare paper, or a line indented first)
  or scratched out, as they are in life.
- **A tool in the hand** (built). A `Smudger` is to a `Tool` what a `Stick`
  is to a medium: the tool as it is held, with what it carries, which it
  keeps from one pass to the next until `clean()`. It lands with its load,
  and it takes less the fuller it is and nothing once it is full: a tone
  can be rubbed again and again without wearing away, no pass leaves a pale
  dot where it lands, and a stump loaded on a dark patch draws a tone on
  bare paper, which takes the dust off it readily (one load lasts thirty or
  so of its widths). A bare
  `Tool` is clean each time it is put to the paper and takes away what it
  still carries; `Sheet.rub` keeps the load across its own passes either
  way.
- **Fixative** (built) acts on the whole sheet. It does not move the dust into
  `held`, which would mix a light pastel into the dark one under it: the dust
  stays in its two layers, and `fixed` records how much of each is set (it
  goes under with its dust when a new pass covers it). Dust that is set is
  lifted only as held pigment is, is not pressed in or stirred, keeps loose
  dust above it from being pressed into the paper,
  scatters a little less light (fixative darkens), and takes half the room in
  the tooth, so that more can be laid over it. What is laid afterwards is
  loose.

## One step of a stroke

A stroke is sampled along its path as in oilpaint; pressure, the width of
the tip and its tilt can vary along it (speed is still to come). A width
that changes draws a stroke that tapers or recedes in perspective; a tilt
that changes turns a stick from its point onto its side as it goes.

The tip lays as it slides, so a pixel gets a pass's worth from all the
places of the tip that cover it. Under the middle of a stroke that is the
whole tip going over it; where the tip lands, the paper under its back half
would get less, from nothing at its back edge, and the same where it lifts.
That is not how a stick behaves: it is put down with all of its face
bearing and lifted the same way, and the end of a stroke is the outline of
its tip at the full tone of the stroke (the hand makes it lighter there by
bearing less, and a shading does). So where the tip lands and where it
lifts, the paper it covers is laid as though the tip had slid onto it from
behind and off it ahead, and a stroke is as strong at its ends as in its
middle, a short one (a head, a hand) as a long one, and a dot is one pass of
the tip. The first build left this out: the ends of a stroke faded out over
a whole width of the tip, and a short stroke of a broad tip reached a third
of the tone of a long one.

Where a stroke turns, the tip turns with it, and the part of it on the
outside of the turn slides further over the paper than its middle, the
inside less: each part lays as far as it slides. That matters for a broad
tip: a band 90 pixels wide laid by the side of a stick round a quarter
circle of radius 150 would have 1.10 of its middle's tone on the inside
and 0.84 on the outside, had each part laid as far as the middle went; it
has 0.95 and 0.96. No part of a tip lays more than a pass's worth on a
pixel in a step, though: where the stroke turns so sharply that its far
end would sweep further than that (under a sixth or so of the half-width
of the mark: 7.5 pixels for that band, under half a pixel for a point 4
wide), the step is laid in parts, the tip turned bit by bit across the
angle, so that what it sweeps is spread along the arc. And a tip is the
same turned half round, so where a stroke doubles back on itself the hand
draws it back without turning it. (A part was laid as far as it slid,
however far: at a cusp tens of passes in a step, which pressed in more
dust than there was.)

A stroke ends round, in the outline of its tip, which reaches half the tip
past the ends of its path; or flat, cut square across at the ends of its
path, as the end or the side of a stick drawn and lifted cleanly leaves it,
so that a flat stroke is exactly as long as its path. A shading's strokes
end round half a tip short of the edge of the area, or flat at it.

At each step, for each pixel under the tip:

1. **Reach.** Pressure, the softness of the medium and the give of the paper
   set how far down the tooth the tip gets: to the ground of a pixel, which is
   its tooth, raised towards the peaks as far as it has been pressed flat, and
   no higher than the floor of a groove; and from where the tip rests, on
   the highest of the paper it covers, down into a groove that all it
   covers is in. A tip is flat across with a
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
   medium's `binding`, and it is what the tip loses. Of the share that is
   held, a waxy medium lays as a film on top whatever met dust and not the
   paper.
5. **Working what is there.** The tip presses some of the dust under it into
   `held`. At the end of the pass its own dust (or paste) and its film
   settle on top of what was there, which goes `under`: all but a little of
   it once what is new is thick enough to cover it. The film and the paste
   bond part of the loose dust they come to lie on.
6. **Wax.** A waxy medium films over the tooth, which then takes less
   (resist). How much less depends on what is laid over it: a dry medium,
   which the tooth has to scrape off, hardly takes at all; a waxy one takes a
   little; a greasy paste such as oil pastel sticks to it. It also depends
   on how thick the film is, and not alike for both: wax goes on taking wax,
   less and less, until the film is `WAX_FULL` thick, so that light layers of
   coloured pencil build; but a dry medium has no tooth left to scrape it
   off as soon as the wax covers the tooth, which is the thickness at which
   any layer covers what is under it (`COVERS`). One firm line of crayon is
   that thick, so charcoal shaded over it skips it: wax resist. Pressed hard, the tip polishes the film: it raises `press`,
   which levels the tooth so that every part of the pixel is touched, and it
   drags some of the waxy layer along and lays it back down, valleys
   included. That closes the white specks (burnishing). A film that lies
   on dust is wax as well, and the tip drags it as much as it drags what
   the paper holds and presses it down into the tooth: one firm layer of
   the blender over three of a dark coloured pencil laid on near-white
   pastel works a fifth of the dark that lay on the dust into the tooth,
   and drawn from such a patch out over the paper it smears 0.13 of the
   dark next to the edge, as it smears 0.12 out of the tooth. (It took
   only what the paper held, and the film's wax, which is counted with
   that, without its pigment: a hundredth of the dark, and 0.08 next to
   the edge.) It closes the white flecks as it does over the paper, by
   polishing. It sets in by
   degrees, between `BURNISH_SOFT` below `BURNISH_FROM` and as far above
   it (from 0.4 to 0.8), since the wax does not give way at one pressure
   everywhere: in there a tip polishes as the square of how far it bears
   past 0.4, and beyond it in proportion to how far it bears past 0.6. (It
   began sharply at 0.6, and the strokes of a shading near there, each a
   little firmer or lighter than the next, polished or did not, which left
   the area polished in streaks: over two layers of a dark coloured pencil
   at 0.6, `press` smoothed over the grain spread by 0.78 of its mean, and
   now by 0.58; of an oil pastel 1.08, now 0.73.) How much a tip polishes
   goes with how slick the film is, and it takes up `BURNISH_LIFT`
   times its polish of what the paper holds. A colourless blender is made
   for this: soft clear wax and nothing else, it melts into the film
   (`Medium.melts`), so it polishes a thin film as if it were slick, takes
   up `MELT_LIFT` times its polish and lays it back within about its own
   width: one firm layer of it stirs about half of what is under it into an
   even film. On three layers of red at 0.5 on cartridge paper,
   a tenth of the pixels still show the paper; one layer of the blender at
   full pressure leaves a fifth of those (two layers a twenty-fifth), the
   chroma of the patch goes from 34 to 59 (69), its lightness from 67 to 57
   (54), it is half as grainy, and it has 0.48 (0.80) of the gloss of
   polished wax; at a pressure of 0.7 the blender closes two fifths of the
   flecks, and at 0.5 a twentieth. (In the first build the blender was a
   coloured pencil without colour: one firm layer left two fifths of the
   flecks.) Two colours laid over each other are stirred into one, and
   a blender drawn from a dark out over bare paper drags it along: a
   quarter of the dark's pigment lies next to the edge, a thirteenth a
   tip's width out, and under a hundredth three widths out. A dark over a light is a
   matter of room and of wax, not of order: on cartridge paper two layers
   of a dark teal at 0.6 over four of lemon at 0.4 take half of what they
   take on bare paper and go two thirds of the way to the tone of the teal
   alone; on a paper with a shallow tooth (depth 0.4), which those four
   layers fill (the fourth takes a fifth of the first), the teal takes a
   fifth and goes a third of the way, and over one layer of lemon it takes
   over half. There the darks go first.
7. **The tooth.** A dry tip pressed hard (harder than `BURNISH_FROM`, and
   setting in by degrees, as burnishing does) flattens the tooth itself,
   and raises `press` too: the hardest lead by `FLATTEN` in a pass at full
   pressure, a softer one as the cube of its hardness (it gives before the
   paper does), and one that crumbles to dust as the square of its binding
   (it bears on the paper only with what holds together). Three firm layers
   leave the tooth flattened by 0.88 under
   a 4H, 0.74 under an HB, 0.37 under a 2B and 0.03 under a 6B; compressed
   charcoal and hard pastel flatten it by a few hundredths, vine charcoal and
   soft pastel next to not at all. A flattened tooth is touched all over, and bites
   less: a dry medium lays `FLAT_BITE` less on it in a pass (the tooth is
   what scrapes it off its stick). So a light layer of a soft pencil over a
   patch a hard one flattened lies on four fifths of it where it marks half
   of a tooth left alone, and far more evenly, for about as much graphite.
   The floor of a groove, where the point that pressed it took the peaks
   off, is as flat, and bites as little.

A stump, a finger or an eraser runs the same way but lays nothing of its own.
Where it bears, it lifts a share of the `loose` pigment (and, for an eraser,
of the `held`, less where wax holds it; paste as dust if it smears, as held pigment if it does not;
what is set and a film as held pigment), carries it, and lays a share of what it carries back down
further on, as dust spread over the whole pixel. It presses some dust into the paper and pushes the dust on the
high ground into the low, so `cover` rises and the white specks grey over. An eraser lays nothing back.

How far a tool carries what it moves is in two parts, and most of it goes
a short way. As strongly as it smears, a tool takes the dust under it onto
its face and lays it back within a few pixels (`SPREAD` times over each of
its widths): a grain is taken up and put down several times as the tool
goes over it, and ends a fraction of the tool's width from where it lay.
That is what evens a rubbed tone, faster than the pressing in, and it is
what blends two colours: a finger rubbed to and fro across the join of an
orange and a blue pastel leaves a passage from the one to the other about
a width and a half of the finger wide (from a quarter of the blue to three
quarters, after 18 passes; two widths after 72), with a tenth of the other
colour two widths out; and of what a stump drags out of a tone onto bare
paper, two thirds lies within one width of the edge. What a tool lifts into
its load is the small share that goes far, and a stump or a finger lifts
little (`Tool.lift_loose`, a twenty-fifth of the dust in a pass): the trail
of that stump is a fiftieth of the tone three widths out and goes on,
fainter.

The load is what a tool tones with, and two rules make a stump that has
been used for long a different thing from one dragged once through a tone.
A stump or a finger holds so much (`Tool.holds`, a volume for each pixel
of its face) and lifts the less the fuller it is: it fills slowly, over a
hundred widths and more of rubbing in a tone, and once full it takes
nothing more. And it gives the dust of its load to bare tooth only
(`hunger`): where the paper carries nothing it lays `Tool.lay` of its load
over each of its widths, less where pigment lies already, and nothing once
that covers the tooth (`COVERS` thick, on the covered share of the pixel).
So a full stump leaves a tone as it is, however often it goes over it, and
a finger loaded with one colour leaves a tone of another clean, where an
exchange both ways would bring its colour all over it; but on bare paper
the same stump draws a tone of its own: loaded on a patch of 6B, 0.72 of
the paper's tone where it lands, 0.83 eight of its widths on, 0.91 after
sixteen and 0.96 after twenty-four, by when it has laid nearly all it
held. (In the first build it laid 0.59, 0.92, 0.98 and 0.99: darker where
it landed, and spent sooner.) A stump dragged once through a tone has
hardly begun to fill, and leaves little beyond what was on its face. A
full stump also lays dust on whatever bare paper it strays onto at the
edge of a tone, as a dirty stump does. Paste sticks to any paper, bare or
not, so what a finger has taken of an oil pastel it lays back as it goes.
Wax is as slick to the dust a tool brings as
to a dry stick (`RESIST_DRY`, where the wax covers the tooth): a stump
loaded with charcoal and dragged across one firm line of crayon leaves on
it an eighth of what it leaves on the paper either side, and the line
keeps its tone; what the wax refuses stays on the tool, which carries it
on. The paste of an oil pastel sticks to wax, on a tool as on a stick. And
a tool slides on a film of wax laid over dust, as on dust that fixative has
set: the loose dust in the layer underneath is stirred up and lifted only
as far as what is set or a film on top does not cover it (all of it, under
a layer of those `COVERS` thick).

The wax of a pixel is in what the paper holds there, in the films and in
the paste, and it goes with them: a tool that takes a share of those takes
that share of the wax, carries it, and lays it back with its paste. So a
finger that smears an oil pastel out over bare paper leaves a smear as
waxy as the pastel (three tenths of its volume), and as slick to charcoal,
and leaves what it rubbed no waxier than it was; and what an eraser or a
knife takes off, wax included, is on the tool. Wax is conserved as pigment
is: on the sheet, or on the tool (`Smudger.wax`). (In the first build a tool lifted two thirds
of the dust into its load in a pass and laid it back over several widths:
a rubbed join was one even mix over all that was rubbed, and the trail of
a stump on bare paper halved every four widths.)
Like a tip, a tool is put down with all of its face bearing and lifted the
same way: where it lands and where it lifts, the paper it covers is rubbed
as though the tool had slid onto it from behind and off it ahead, so a pass
is as strong at its ends as in its middle, out to the outline of the tool,
and a dab is one pass of the tool on the paper under it. The passes of a
rub of an area therefore stop half a width of the tool short of its edge,
as the strokes of a shading do, and the area is rubbed up to its edge and
no further. (It used to rub
only as it slid on: at the ends of its path an eraser took two thirds of
what it took further on, and a dab a quarter.)
What a tool carries is in three parts: what it has lifted, which it lays
back at its own rate; the dust on its face, which it lays back within a few
pixels (in each step it shares out among the pixels under it what was on
its face as the step began, each by how it bears there); and the held
graphite it is pushing along, likewise. All three stay on a tool
in the hand (a `Smudger`) when it is lifted, and it lands with them; what a
bare tool still carries at the end is counted and returned. Either way
nothing is lost.

These are numba kernels under the same rules as oilpaint's (no `fastmath`, no
`parallel`), so a drawing repeats exactly, and what makes them fast changes
no result: `tools/fingerprint.py` hashes all that a suite of drawings leaves
on its sheets, and a change that is not meant to alter the picture leaves
every hash as it is. Their constants are in
`fusain/_engine.py`. They were set so that tones land near these figures,
which are estimates from experience, not measurements of real drawings: on
cartridge paper and without its shine, one even layer of HB reflects about
0.8 of what the paper does at a light pressure and 0.4 at full pressure; 6B about 0.5 and 0.13; 4H
at full pressure about 0.75; and light layers of HB level off near 0.5.

## Rendering

Per pixel: Kubelka-Munk reflectance of the `loose` layer over the `under`
layer over the `held` layer over the paper's own colour, on the covered share
of the pixel; the rest shows bare paper. A film is in the layer it was laid
in, so a crayon drawn over charcoal dust is seen over it.
By default a fairly high lamp shows the tooth faintly, as on a scan; a raking
light brings it out, and `lighting=False` renders the sheet flat.

Three things shine, each where it lies flat, and `Sheet.shine(of)` says how
much of each pixel does. The sheet is seen from straight above, so the
mirrored light is strongest with the lamp overhead and weakest under a
raking one; it is added to what the pigment sends back.
`render(gloss=0)` leaves all of it out, as a scanner does.

- **Graphite** shines like a metal. A film of it that the paper holds mirrors
  the lamp, the more the thicker it is (a heavy layer of a soft lead shines
  fully, a light layer of HB hardly at all), and matte dust lying over it
  hides it. On a tooth pressed flat the graphite lies flat, and a film two
  fifths as thick shines as much: three firm layers of 4H shine 2.5 times as
  much as the same graphite would on a tooth that had kept its shape (0.31
  against 0.12 of the most), and of HB 1.4 times; six of HB shine a third as
  unevenly. A soft lead shines by how much it lays, a hard one by how flat
  it has pressed the paper. The light is white, in a broad lobe, since
  graphite is satin and not a mirror. Under the default lamp, three firm
  layers of 6B reflect about 0.10 of what the paper does, against 0.06
  without the shine: graphite never looks as black as charcoal.
- **Wax** has no sheen of its own: it is its polish that shines. The gloss is
  how slick the film of wax is at the pixel (as for the resist) times the
  square of how far the surface is pressed flat (a film half polished still
  scatters most of the light), so a coloured pencil laid lightly has
  none, one burnished with a blender eight tenths of the most, and two
  layers of crayon from none at a pressure of 0.3 to three quarters at full
  pressure. It counts
  wherever the wax lies: a film laid over dust is on top and shines; what
  the paper holds shines through whatever dust lies over it, as graphite
  does. The light is white, less than half as strong as graphite's, and in a
  narrower lobe, since a film of wax polished flat is smoother than one of
  graphite flakes: it falls to half about 15 degrees from where the lamp is
  mirrored, and to a fifth at 23 degrees, which is how far from the eye
  above the sheet the default lamp is mirrored. So with the lamp overhead
  burnished wax shines, a soft white gloss, and under the default lamp a
  burnished dark stays deep, as it does in life away from the glare: four
  firm layers of a dark coloured pencil burnished with the blender go from
  L* 7 to 10 under it and to 19 overhead, and the gloss that the firm
  layers alone show varies from place to place by a third of what it did.
  (In the lobe it had first, wider than graphite's, the gloss greyed that
  dark to 18 under the default lamp, and followed the polish stroke by
  stroke.)
  How narrow is assumed, not measured: polished wax is taken for a smooth
  dielectric whose facets lean about 0.16 on average (a semi-gloss
  finish; graphite's, in its lobe, about 0.38).
- **Ballpoint ink** bronzes where it is heavy: from one and a half firm
  passes thick, fully at four and a half, so eight layers of firm hatching
  bronze half-way and a single line not at all; what is drawn over the ink
  hides it. Ink mirrors most the light it absorbs most (as the square of its
  absorption), so the bronze of blue ink is copper and that of red ink
  green; but a black ink is a blend of dyes, and bronzes as the one that
  bronzes most, its violet: a pen can say the colour (`Pen.bronze`), and
  `pens.BLACK` bronzes a red between copper and violet, where it bronzed
  grey. Each pixel keeps the colour of its ink's bronze, mixed as the inks
  are. A film of dry ink is smooth and its lobe narrow: the bronze shows
  with the lamp near overhead, a tenth as much under the default lamp and
  not under a raking one.

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
- **Too crude:** the stump blurred to a clean grey band. The stump now built
  bears unevenly across its width, like a lead, and leaves streaks along its
  way.

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
on at a steady rate, not something the paper scrapes off. It has its own
kernel and its own layer:

- **The ball** is hard and small, so even a light touch reaches the valleys of
  the tooth: a light line is still a line, not specks on the peaks. It is
  round, so a light line is narrower than the ball. Where it lands and
  lifts, the paper it meets gets what it would have had if the ball had
  rolled onto it and off it, as under a stick: a line is as strong at its
  ends as in its middle, and a dot as dark as a line.
- **The ink** is laid at a rate that does not depend on the tooth or on how
  much is there already, only on the pressure (the lightest touch lays about a
  seventh of the firmest) and on a slow unevenness along the line. It is a
  clear dye: lines that cross and layers of hatching darken towards black.
- **It is a stain** in the paper, rendered under whatever the paper carries,
  and no tool touches it: an eraser takes the pencil sketch and leaves the
  ink. A pencil draws over it as on bare paper.
- **The ball skids** on a thick film of pigment and on wax, and lays little
  there; over a light pencil sketch it writes as on bare paper.
- **The ball presses a groove** (`dent`), as a stylus does but less deep
  (`BALL_DENT`), in proportion to the pressure. A soft pencil shaded lightly
  over a pen line fills it nearly as the paper beside it if the pen was
  light (a pressure of 0.2), a quarter at 0.6 and hardly at all at full
  pressure; over three layers of light hatching, which pressed the paper
  down all over, it goes down with the paper and lays a tenth more than on
  paper left alone, and over firm hatching three fifths of that. The
  ball itself goes down into a groove: it is as small as what made it, so
  one pen line crosses another as if the paper were flat.
- **The ball gathers ink.** The share `BALL_GATHER` of what it lays stays on
  its rim as it rolls, up to a drop (`BALL_BLOB`), and a turn brings it
  round onto the paper: nearly all of it when the line turns back on itself,
  about two thirds at a square corner, a little at a time on a gentle curve,
  where it does not show. A drop too small to show stays on the rim until
  the line has turned far enough to bring round one that would
  (`BALL_SHOWS`): a gently curving line had a drop at every point, which
  took a third of the pen's time. The paper takes of what comes round as
  much as the ball bears on it (the share of a firm pass the line lays
  there, from `BALL_LIGHT` to all of it), and the rest stays on the rim
  for the next turn: a light line blots as lightly as it is drawn, where it
  had blobs as big and as dark as a firm line's. So a sharp turn leaves a
  blob, the bigger the longer the line before it (after 140 pixels at a
  pressure of 0.5, a third as much ink again at a square corner as on the
  line beyond it). A blob is a round dot a little wider than the line,
  thickest at its middle, and it is ink the pen lost like the rest:
  `stroke` returns it with the line. How much a pen gathers is its own
  (`Pen.blots`: 0 for one that never blots, more than 1 for a leaky one),
  apart from how freely its ink flows (`Pen.flow`, which darkens the line
  too).
- **A pen put to the paper has ink on its ball already.** A pen in the
  hand (`HeldPen`) lands with what its last line gathered and did not
  leave: of that, where it lands, a share drawn at random comes round,
  mostly next to none, so that one line in four or so starts with a dot
  that would show on a firm line, and a few with a blob; the rest comes
  round at the first sharp turn. It keeps the ink on its ball until it is
  wiped; what it gathered is on the paper or on its ball. A bare `Pen` is
  picked up afresh for every line: it lands with as much as such a share
  of the most its ball holds, which it leaves where it lands, and takes
  away what it gathers.
- **Heavy ink bronzes** under a lamp: see Rendering.

## Sketch of the API

```python
from fusain import Sheet, Smudger, papers, pencils, pastels, tools

sheet = Sheet(1200, 800, paper=papers.CARTRIDGE, seed=1)
sheet.stroke([(100, 300), (300, 220), (500, 320)], pencils.HB, pressure=0.4)
sheet.shade((100, 360, 500, 520), pencils.HB, pressure=0.4, angle=30)
sheet.stroke([(100, 560), (500, 550)], pencils.B4, tilt=1)   # the side of the lead
sheet.stroke(path, pastels.soft("#c0392b"), pressure=(0.2, 0.8), tilt=1)
sheet.stroke(path, pencils.B2, width=lambda u: 8 - 6 * u)  # tapers
sheet.stroke(path, pastels.hard("#2c3e50"), width=6, ends="flat")  # cut square
sheet.indent(path, pressure=0.7)     # a stylus: a groove that a tone skips
sheet.shade(area, pencils.BLENDER, pressure=1.0)   # burnish coloured pencil
sheet.smudge(path, tools.STUMP)
sheet.erase(path, tools.KNEADED)
sheet.erase(path, tools.KNIFE)       # scratch a light out of wax
sheet.rub(area, tools.FINGER)        # a whole area, to and fro
stump = Smudger(tools.STUMP)         # in the hand: it keeps what it picks up
sheet.rub(area, stump)
sheet.smudge(path, stump)            # loaded, it tones bare paper
stump.clean()
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
