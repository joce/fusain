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
| `crush` | How far the paper's own tooth has been crushed here (0 to 1), its peaks pressed down towards its valleys: by a dry tip pressed firmly, the more the harder the tip. It is the paper's: it stays whatever is laid or lifted afterwards, until a blade or grit cuts the surface of the paper away. |
| `polish` | How far the wax that lies here has been polished level (0 to 1): by any tip pressed firmly on a film of wax, and by a blender. It is the wax's: it goes as the wax that filled the tooth is taken off. |
| (`press`) | Not stored: what a tip meets, worked out from the two (`Sheet.press`, `crush + polish - crush * polish`). A tooth pressed flat is level: a tip touches all of it, as it would its peaks, and in the relief its peaks have come down to its valleys. |
| `dent` | How far the paper has been indented here: the depth of a groove below the peaks of the tooth, in depths of the tooth of a paper of depth 1. By a stylus, and by the ball of a pen. |
| `cover` | Share of the pixel that carries pigment (0 to 1); the rest shows bare paper. |
| `held` | Pigment fixed to the paper: absorption K over 24 spectral bands, scattering S, volume, how much of the volume shines (graphite) and how much is wax. The wax counts all the wax at the pixel, wherever it lies: in what the paper holds, in a film or in paste. |
| `ink`, `inked` | The ink of pens, a stain in the paper under everything else: K, S and volume, and the colour it bronzes (linear RGB, times the volume); and the share of the pixel it is on. Allocated when a pen first draws. |
| `under`, `loose` | Pigment lying on what the paper holds: K, S, volume and sheen, in two layers. `loose` is on top; what a new pass covers goes `under`. Most of it is dust. |
| `bonded` | Of what lies in each of the two layers, the volume that is not loose dust, of three kinds: `fixed`, dust that is set, by fixative or under a film; `film`, what a medium laid on the dust and not on the paper, a waxy one wherever dust lay and a dry one on a bed of it; `paste`, what the paper does not hold of a greasy medium. |

That is about 0.38 KB per pixel (95 numbers of four bytes; 94 while one
`press` stood for `crush` and `polish` both), a little more than oilpaint. As there, storing summed
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
  is as fast as what the paper holds: only a hard eraser takes it off.
- **A bed of dust.** A dry medium lies on what it meets too, but its own
  crumbs are not in its way. How much of a mark the paper holds
  (`binding`) is how much it holds on the tip's own ground: bare paper, or
  what the tip has laid itself, a film with its crumbs on it, through
  which it reaches the paper. What keeps it from the paper is dust beyond
  that: more dust, loose or set, than the tip's own mark would have for
  what is fast at the pixel (`(1 - binding) / binding` of what the paper
  holds, of a film and of paste), by more than a whisper (`BED_FROM`).
  That much is a bed between the tip and the paper, and three things
  follow from it.
  The tip rides on the bed as far as it covers the paper (all of it once
  the bed is `COVERS` thick), and there what the paper would have held of
  its mark is a film on the dust, on top with the dust of the pass, as a
  waxy medium's is; but it has no wax, and binds none of the dust. A film
  is what such a mark has to be: it is seen over the dust; it counts in
  the tooth as what the paper holds does, so that layers of a pencil over
  pastel build and level off as they do on paper; and the tip bears on
  it, so the passes that follow do not press it, or the dust under it,
  down into the paper. (Left loose with the dust, a mark could not be told
  from the bed it lay on: it took no room, piled up without end where set
  dust kept it from being pressed in, and elsewhere was pressed under the
  pastel, so that thirty layers came out lighter than eight.)
  A bed is not in the tip's way, whether it ploughs through it or rides
  on it: it takes none of the room in the tooth, where it took all of it
  and a tooth full of pastel dust took next to nothing of a pencil. And
  where the tip rides on the bed, the paper under it is out of its reach:
  what the paper holds there is not in its way either, and what fills the
  tooth for it is what lies on the bed with it, films and their crumbs.
  Where it reaches the paper, through dust too thin to cover it, what is
  fast there fills the tooth as before, with crumbs such as its own.
  And dust scrapes less off a tip than paper does, loose (it rolls under
  it) or set: riding on a bed, a pass lays `BED_BITE` less.
  So a pencil marks pastel, a little weaker than bare paper, and a firm
  stick marks over the dust of a crumbly one, not the other way about:
  vine charcoal over pastel is dust on dust, and finds the tooth full.
  On cartridge paper, over two layers of a pale soft pastel laid at 0.6, a
  2B line drawn at 0.6 lays 0.78 of what it lays on bare paper and is 0.80
  as dark against its ground (0.02 and 0.01 before, the little there was
  lying under the dust), an HB 0.77 and 0.72, a charcoal pencil 0.77 and
  0.91; over the same fixed, 0.79 and 0.81, 0.78 and 0.74, 0.78 and 0.92
  (0.15 and 0.03 before for the 2B); a 2B shaded at 0.5, 0.82 and 0.70.
  All of the line's graphite lies on the pastel, where 0.15 of it did,
  its crumbs; over a light layer of pastel 0.8 of it, and over the dust of
  vine charcoal 0.7, the point reaching the paper through the rest.
  A film of wax holds: it sets the dust under it, a tool slides on it and
  lifts it only as it lifts what the paper holds. The film of a dry medium
  has nothing to hold on to but the dust it lies on, and is as fast as
  that dust: where a pixel has no wax, a tool lifts a film and spreads it
  as it does the dust, as far as the dust of the two layers is loose (all
  of it on loose pastel, next to none on pastel that is fixed), and what
  it spreads is dust where it is laid; and the film is not slick, so the
  tool reaches the dust under it. Of a 2B line on loose pastel a kneaded
  eraser takes 0.79 in a pass, with the pastel under it (it takes 0.58 of
  the pastel beside the line, and 0.54 of what is under it), a quarter of
  a line on bare paper or on fixed pastel; a vinyl eraser 0.93, and 0.81
  of a line on bare paper or on fixed pastel; a finger drawn three times
  across it carries off a fifth of its graphite and leaves it 0.67 as
  dark (0.07 and 0.93 on bare paper, 0.03 and 0.96 on fixed pastel).
  (Lifted as what the paper holds, the film was a lid: the kneaded eraser
  took 0.36 of the line and left 0.80 of the pastel under it, where it
  left 0.42 beside it.) The film has no wax, so it leaves no stain: six
  passes of a vinyl eraser take all of the line, on pastel loose or fixed
  as on bare paper, and the point of a knife drawn along it leaves a
  hundredth of what lay in its way.
  A medium laid over its own layers, or on a ground no dustier than its
  own mark, meets no bed and draws as it did, to the bit: graphite layer
  on layer, of one grade or of several, pastel over pastel, fixed or not.
  Two limits. What the paper holds fills the tooth for every medium
  alike, so pastel rubbed into the paper takes a pencil only while dust is
  left on it for the point to ride on: over those two layers blended with
  a finger a 2B line is 0.77 as dark as on bare paper, blended again
  crosswise 0.23; and a stick that leaves more crumbs of its own takes
  blended pastel for a ground like its own, a tooth that is full: a
  charcoal pencil over the pastel blended once, 0.21. A film that a later
  pass buries is one mixture with the dust of its layer, so a pencil over
  pastel levels off greyer than on paper, and a second layer of a 6B at
  full pressure, thinner than the first, leaves the patch a little lighter
  than the first did (0.24 of the pastel's tone, then 0.29 and 0.31).
  A film of graphite shines under the lamp as what the paper holds does
  (see Rendering).
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

A tip meets the surface in two ways. Where a whole area has been pressed
flat (its tooth crushed, its wax polished, or both: `press`, below), the
tip that comes next rides on it: what
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
dry medium as little; and the whole floor of a shallow groove bites as a
tooth flattened as far as the groove is deep, as crushed paper does (its
valleys bit as on fresh paper, so that paper dented evenly took 1.23 to
1.43 times what paper crushed as far takes, and 1.16 to 1.24 now, the
rest being that a stylus takes the peaks off where a crush lifts the
valleys too). Both show in the
relief the lamp lights. There a flattened patch has been pressed down: its
peaks come down towards its valleys, by `press` of their height, so that
pressed flat it is level with them, a little below the paper around it,
and a valley the tip never reached is not left a pit. (They were raised to
the peaks, as for a tip, and such a valley was left a pit as deep as the
tooth: six firm layers of 4H left a twentieth of the patch more than a
quarter of a tooth below the rest, and under a raking light the patch
looked as rough as the paper left alone; now none, and a seventh as rough.)
A groove is a line of shadow under a raking light.

### Pressed flat: the paper's crush and the wax's polish

Two things level a tooth, and they are not the same thing, so the sheet
keeps them apart.

`crush` is the paper's own. A dry tip pressed firmly crushes the fibres:
the peaks of the tooth come down towards its valleys. A tip crushes as far
as it is hard, dry and all of a piece (`FLATTEN`, in "One step of a
stroke"), and it does so whatever lies on the paper: a hard lead crushes
the tooth through a layer of coloured pencil as it does through its own
graphite. Nothing that is laid or lifted afterwards changes it. No stroke,
stump, finger, rubber eraser or fixative gives a crushed tooth back; only
what cuts the surface of the paper does, a sand eraser or the point of a
knife, by the share of the surface it cuts.

`polish` is the wax's. Any tip pressed firmly on a film of wax polishes it
level, as far as the film is slick (`WAX_FULL` thick) or the tip melts
into it (a blender, a pale pencil): a waxy tip its own film, and a dry one
too, which raises both where it is pressed hard on wax, the polish of the
wax and the crush of the paper under it. The polish goes with the wax.
Polished wax keeps the tooth level as far as it fills it, `WAX_FAST` thick:
where a tool takes wax off, the polish left is at most what the wax left
is thick over that, and with the wax gone it is gone. That is worked out
for each pixel once the tool has done all it does there, with the wax and
the cover it leaves (it drags wax off and lays dust after it has lifted).
A tip raises the polish only where it leaves wax: one that melts into the
film polishes it however thin it is, but over a tooth so full of a dry
medium that it lays nothing there, nothing. Where a tip that does not melt
drags wax off a pixel, the polish goes with it as under a tool. A blade takes the polish with
the surface it cuts. Fixative leaves both as they are.

Each is read for what it is:

| What reads it | Which |
| --- | --- |
| The tooth a later tip meets, and the ball of a pen | Both: `press = crush + polish - crush * polish`. Polished wax levels its share of the relief that the crush left. |
| How little a level tooth bites a dry medium (`FLAT_BITE`) | Both, as `press`: wax polished level and paper crushed level have as little to scrape a stick with. |
| The relief the lamp lights | Both: the peaks come down by `press` of their height. |
| The sheen of graphite, which lies flat on a level tooth (`SHEEN_FLAT`) | Both, as `press`: it lies on what the tip met. |
| The gloss of wax | `polish` alone: wax laid lightly on a tooth that a hard lead crushed is not polished. |
| How wax holds against an eraser (`POLISH_HOLDS`) | `polish` alone. |
| What a rubber eraser gives back as it takes the wax | `polish`, and never `crush`. |

`Sheet.press` is what a tip meets, worked out from the two; it is read,
not written.

(There was one `press` for both, and where a pixel held wax it was taken
for the wax's polish. So the crush of a hard lead under or over coloured
pencil went with the wax when that was erased, wax laid lightly over a
crushed tooth shone and held against an eraser as if it had been
burnished, and where a blender had polished further than thin wax holds
up, the rest stayed when the wax was gone. The second number costs four
bytes a pixel, a hundredth of the sheet.) Measured on cartridge paper:
three firm layers of 4H crush the tooth by 0.92; three layers of a dark
coloured pencil over them, rubbed out with a vinyl eraser, leave the crush
as it was to the bit, and a light layer of HB shaded there touches 0.94 of
the pixels, as it does on the crushed tooth alone (the crush fell to 0.05
with the wax, and the HB touched 0.69; fresh paper 0.68). Six light layers
of crayon over such a tooth have no gloss, and a rub of a vinyl eraser
leaves 0.26 of them, as of the same layers on fresh paper, 0.30 (they had
0.64 of the gloss of polished wax, and a rub left 0.69). After a blender
pressed hard over a very light layer and six rubs of the eraser, no pixel
keeps more polish than the wax left on it holds up, to within the 0.002
that the eraser drags off after it has lifted (a fifth of them kept 0.01
or more beyond it).

**Each goes to a level set by the load.** A pressure flattens wax and
fibres as far as that pressure can, however often it is repeated. A tip
that bears with firmness `f` (`firmness`: nothing up to a pressure of 0.4,
1 at full pressure, as burnishing sets in) has a level for each:

- for the polish, `min(BURNISH * f, 1)` of how slick the film is: a slick
  film is polished flat from a pressure of 0.87 on, to 0.75 at 0.8, 0.42 at
  0.7, 0.19 at 0.6 and 0.05 at 0.5;
- for the crush, `min(crush * f, 1)`, where `crush` is what the tip's
  hardness makes of `FLATTEN`: at full pressure a 4H crushes the tooth
  flat (from 0.88 on), an HB to 0.38, a 2B to 0.18 and a 6B to 0.014. A
  softer tip gives before the paper does, however long it is rubbed.

These are the numbers that `BURNISH` and `FLATTEN` already were: what one
pass's worth of sliding under that bearing added to `press`, which is now
where the bearing stops. A pass brings the press towards its level: each
part of it closes a share of what is left of the gap, `SETTLE` (2) times
the share of a pass that it is (and as many times more as `BURNISH * f` or
`crush * f` exceeds 1: the hardest leads are there at once). So the first
of a pass over a tooth left alone raises it twice as fast as it did, and
the nearer its level, the less: a whole pass ends about where it ended
when `press` added up, a second closes most of what is left, and a third
changes next to nothing. A firmer pass over a lighter one raises the
press to its own level; a lighter one over a firmer one leaves it as it
is. What a firm tip smears of the wax (`BURNISH_LIFT`) is work done on
every pass, and is as it was.

What presses flat is the tip, not the grain of its lead. A lead bears up
to a third more or less in bands along the stroke (`STREAK`), another way
in every stroke, which is the grain of its line; a tip is one hard thing,
and what it presses flat it presses as far as it is itself pressed, at
that place of its face (its pressure, less towards its rim). So the level
goes by the tip's bearing with its streaks aside, and so does how fast a
pass gets there. Were the level that of the streaks, every stroke would
leave bands polished from a fifth to all the way under one pressure, and
each further pass would raise the paper under its hardest bands: going
over it again would go on polishing, which is what a level is there to
end. (Measured so: one stroke of the blender at 0.7 polished by 0.13, two
by 0.22, three by 0.51.) What the tip smears and what it lays go by the
streaks, as before.

One pass, measured: a tip 6 wide drawn once along each of eight lines, and
the press under its middle (three rows), on cartridge paper; before, when
every pass added, and now:

| Pressure | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
| --- | --- | --- | --- | --- | --- |
| The blender, on six light layers of red | 0.19, 0.11 | 0.40, 0.29 | 0.60, 0.55 | 0.77, 0.81 | 0.88, 0.92 |
| A coloured pencil, on the same (a film two fifths slick) | 0.10, 0.06 | 0.21, 0.15 | 0.33, 0.28 | 0.45, 0.42 | 0.54, 0.50 |
| 4H, on bare paper | 0.07, 0.06 | 0.17, 0.16 | 0.31, 0.35 | 0.44, 0.58 | 0.56, 0.72 |
| HB | 0.03, 0.03 | 0.07, 0.07 | 0.13, 0.13 | 0.17, 0.21 | 0.22, 0.29 |
| 2B | 0.017, 0.014 | 0.037, 0.035 | 0.063, 0.066 | 0.087, 0.103 | 0.107, 0.141 |

From 0.7 up a pass does what it did to within a third either way, and
with one rate it cannot do better than that. At full pressure the weak
bands of a lead crushed less than the tip does, so a dry pass crushes a
third more. Near where a tip begins to press, the streaks took the
hardest bands of a lead past 0.4 where the tip's own pressure is not yet
there: a pass of the blender polishes a quarter less at 0.7, two fifths
less at 0.6, and at 0.5 a third of what it did (0.02 for 0.06). A second
and a third pass of the blender at 0.7
bring 0.29 to 0.37 and 0.40, and three more to 0.41, the level of 0.7
being 0.42 (they made 0.66, 0.87 and 0.96); of an HB at 0.9, 0.21 to 0.26,
0.28 and 0.29 (0.37, 0.55 and 0.97). The blender at 0.7 over the blender
at full pressure leaves 0.928 at 0.929 (it took 0.913 to 0.957); at full
pressure over 0.7 it takes 0.30 to 0.95.

(Before, `press` was the sum of what every pass added, up to 1. Where the
strokes of a shading overlap, a pixel has two or three passes for its
neighbour's one, and was polished two or three times as far. And any
pressure above 0.4 flattened the tooth in the end: six firm layers of HB
crushed it by 0.99, as a 4H does, and twelve layers of a coloured pencil
at 0.6 polished it by 0.82.)

**How evenly a shading is polished.** The spread of the polish over an
area, smoothed over the grain (a box of 3 pixels), as a share of its mean;
one layer at 0.7 and at full pressure on cartridge paper, before and now:

| | Coloured pencil | Wax crayon | Oil pastel | The blender, over three light layers |
| --- | --- | --- | --- | --- |
| Ruled strokes, all alike (`hand=0`), at 0.7 | 0.35, 0.13 | | 0.42, 0.12 | 0.28, 0.06 |
| at full pressure | 0.21, 0.08 | | 0.24, 0.06 | 0.07, 0.01 |
| The hand as it is, at 0.7 | 0.44, 0.35 | 0.60, 0.49 | 0.65, 0.63 | 0.45, 0.40 |
| at full pressure | 0.28, 0.20 | 0.39, 0.29 | 0.44, 0.38 | 0.19, 0.16 |

Two crossed layers at 0.6, the measure of the soft onset: a dark coloured
pencil 0.58, now 0.41; an oil pastel 0.73, now 0.57; at full pressure 0.18
and 0.15, now 0.13 and 0.17.

Under strokes all alike, a layer is now polished about evenly, to within
0.13 at 0.7 and 0.08 at full pressure (it was 0.35 to 0.42 and 0.21 to
0.24: the passes that overlapped added up). What is left with the hand as it is, is the hand's, and it is most of
what there was. Its strokes are firm by 15% either way (`FIRMNESS`), and
near where a tip begins to polish that is the difference between a little
and a lot: at 0.7 the levels of such strokes run from 0.18 to 0.77, a
spread of 0.39 of their mean before anything is drawn (0.49 at 0.6, 0.20
at 0.85, none at full pressure, where all are flat). They are lighter at
their ends (`LANDING`), and they stray (`STRAY_ACROSS`), so that some
paper lies under the rim of a tip and never under its flat, and is
pressed less. With every stroke as firm, an oil pastel at 0.7 comes down
from 0.63 to 0.47; as firm at its ends, to 0.43; and in its place, to
0.13. And for one layer of a medium that does not melt into its film,
the film itself is not yet slick all over: a quarter to three quarters
as slick as it gets, uneven by 0.11 to 0.24, and polished no further than
it is slick. A level does not even out the hand; it stops
the count of passes from adding to it. More layers do even it out, since
each place keeps the level of the firmest pass that went over it: twelve
layers at 0.7 are uneven by 0.10 where one is by 0.35.

**Many light layers fill the tooth; they do not polish it.** Coloured
pencil is built in light layers, which keep the tooth, and it is burnished
at the end, by pressing hard: with the pencil itself, with a pale one, or
with the colourless blender. The model now says the same. Twelve layers of
a coloured pencil at 0.5 polish the wax by 0.05 and have 0.002 of the gloss
of polished wax; at 0.6, 0.20 and 0.03; at 0.7, 0.49 and 0.18: no further
than the firmest of their strokes press a film that is thickening. (They
polished it by 0.29, 0.82 and 0.97, with 0.09, 0.54 and 0.73 of the gloss:
twelve layers at 0.6 were burnished as by a firm one.) What the layers add
is wax, and the thicker the film, the slicker it is and the further a firm
pass then polishes it: one firm layer of the blender over those twelve
polishes them by 0.85 to 0.91, with 0.55 to 0.64 of the gloss. A layer
pressed hard, or the blender, polishes at once what no number of light
layers does. And a layer polishes no further than its film is slick: two
layers of a coloured pencil at full pressure, whose film is thin yet,
are polished by 0.55 (0.77 when their passes added up past that), with
0.18 of the gloss (0.37); a crayon's and an oil pastel's, which are
thicker, by 0.83 and 0.85, with 0.72 and 0.75.

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
| `melts` | How far it melts into a film of wax it is pressed hard on, and stirs it | everything else | a colourless blender; a white or a pale coloured pencil, nearly |

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
  ellipse for every medium.) The hand lays each band beside the last as
  steadily as for the stick's thickness, by which it strays across and its
  band varies, not by its length: straying by a share of the length parted
  the bands where they met, in lines of bare paper +10 to +20 in L* under
  compressed charcoal and soft pastel, and +2.5 at most now. Where they
  meet the bands still double up into a faint dark line (−1 to −5 in L*
  under a natural hand, −3 to −9 with ruled strokes): a light pass lays
  more than its share, and overlapping less would leave a dull line in
  polished wax instead, which goes by the firmest pass.
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
  (`polish`). One rub to and fro of a vinyl eraser at 0.9 takes five sixths
  of one light layer of a dark coloured pencil, four fifths of three layers
  at 0.5, over half of four firm layers and a third of those burnished;
  three rubs leave a twentieth of the firm layers, lighter by then (under
  half of the paper's tone), and a third of the burnished ones, as dark as
  they were. (Four firm layers held as the burnished ones do, a rub taking
  over a third and three leaving a quarter, while their passes added up to
  a polish of 0.84; at 0.8 they polish the wax by half, and hold as wax
  that has half kept its tooth.) A kneaded eraser takes over a quarter of
  the light layer in a rub and a fourteenth of the burnished ones. (Before, a vinyl eraser took nine
  tenths of anything in a rub, and a kneaded one a third.) A tool with
  `scrape` cuts through the wax by that share: a sand eraser
  (`tools.SAND_ERASER`, 0.8) takes nine tenths of the burnished dark in a
  rub, and the point of a knife (`tools.KNIFE`, 1) scratches a line two
  pixels wide back to the paper (it takes all but a hundredth of what lay
  on its line, and the wax with it, so the line takes pencil again). What
  they take off is on the tool, as for any eraser. The lights of a coloured
  pencil drawing are therefore kept (bare paper, or a line indented first)
  or scratched out, as they are in life. Where wax has been polished it is
  the wax that is level, not the paper, and the sheet keeps the wax's
  `polish` apart from the paper's `crush` (see "Pressed flat"). Polished
  wax keeps the tooth level as far as it fills it, `WAX_FAST` thick: as a
  tool thins it below that, the polish goes (it is at most what the wax
  left is thick over `WAX_FAST`), and with the wax all gone the tooth is as
  the paper's own, but for what a dry tip crushed, under the wax or
  through it: that stays, as it does under an eraser on graphite. A sand
  eraser or a knife cuts the surface of the paper as far as it scrapes,
  and takes the crush with it. So a light layer of HB shaded over a burnished dark
  scraped back with a sand eraser or a knife touches 0.69 and 0.67 of the
  pixels there, where it touches 0.68 of fresh paper, and is as grainy;
  it touched 0.96 and 0.998 of them, and lay three or four times as
  evenly, the tooth having stayed level. A vinyl eraser leaves most of the
  wax of a burnished dark, which stays level.
- **An eraser smears wax** (built). An eraser that slides on wax that holds
  against it drags some of it along (`WAX_DRAG` of what it would have
  lifted, had the wax not held: so of a dense or burnished layer, and next
  to nothing of a light one) and lays it back, held, `DRAG_LAY` of it over
  each of its widths, on what is beside it and after it; wax sticks
  wherever it is laid. A vinyl eraser rubbed to and fro out of four firm
  layers of a near-black pencil, burnished, over the bare paper leaves a
  grey halo 14 from the paper's colour (CIE 1976) within a width of the
  dark, 6 at three to five widths and 1.4 at six to nine; not burnished,
  11, 4 and 1.1; of a red, 5. (It left the paper clean.) What it drags is
  on the tool until it is laid, and counted with what it carries.
- **Wax stains the paper** (built). A waxy medium carries its pigment into
  the fibres of the paper: of what the paper holds of it, `STAIN` thick on
  the covered share of the pixel (as far as what it holds is waxy) is out
  of reach of an eraser, which takes the rest; a tool that scrapes reaches
  it as far as it scrapes, cutting the paper. It is a rule on what is
  stored, not a quantity of its own: the fibres take only so much, so it is
  a thickness, and the stain is that of whatever the paper holds there,
  which is one mixture. (A layer of its own would cost a pigment's worth of
  floats a pixel and a layer more to render, and would tell apart only a
  stain laid under another colour, which the mixture blends.) So a tint
  always stays where coloured pencil, crayon or oil pastel has been
  erased: a light layer of a near-black pencil at 0.3 rubbed out with a
  vinyl eraser keeps 0.07 of its pigment on cartridge paper and 0.05 on a
  smooth one, 0.87 and 0.71 of the paper's tone however long it is rubbed
  (it came off to the last thousandth, back to the paper's tone); a red
  one leaves a tint 2.5 and 6 from the paper (CIE 1976). A sand eraser
  takes most of it. Graphite leaves none.
- **A tool in the hand** (built). A `Smudger` is to a `Tool` what a `Stick`
  is to a medium: the tool as it is held, with what it carries, which it
  keeps from one pass to the next until `clean()`. It lands with its load,
  and it takes less the fuller it is and nothing once it is full: a tone
  can be rubbed again and again without wearing away, no pass leaves a pale
  dot where it lands, and a stump loaded on a dark patch draws a tone on
  bare paper, which takes the dust off it readily. What it lifts works into
  its fibres, a deep load that comes back out slowly (`Tool.deep`), so a
  stump that has been used for long stays grey. A bare
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

The tip lays at each sample, and the samples are a quarter of its width
apart (half a pixel for the narrowest): each pixel under a stroke is laid
on about four times, half as often as at an eighth, which they were until
`fusain-6ru` and which took twice the time. The tone is a little darker
where the paper fills up (a 4B pressed hard 0.5 L*, an HB 0.14), the grain
a little coarser, and along a line on paper with no tooth the tone ripples
by 2% of itself, a third of an L*, a step apart; on any paper with a tooth
that is ten times smaller than its grain. A tool that rubs keeps an eighth,
and so does the point of a stylus: at a quarter a rub leaves beads along
its way and a groove ripples.

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

The cut is the line through the end of the path, across the way the stroke
goes there (that of its last two samples). It takes what the tip lays past
it for as long as the tip is within its reach of that end along the path,
half its width and a sample's spacing: a stroke that comes back across the
line later is not cut there. And it takes what lies beyond the stroke, not
the stroke itself. Where the path bends within that reach of its end (a
hook shorter than half the tip, a curl tighter than the tip), the band the
tip drew before the bend lies across the line, and stays: what is within
half the tip's flat of the path at one of its samples within reach of the
end, and behind the line across the path there (ahead of it, for the start
of a stroke). A flat end is then what the tip's width sweeps along the
path up to its end, and no more. A straight end has none of its band past
the cut, nor has a bend half a tip or more before the end, nor a curve no
tighter than the tip: those are cut as they were, to the bit. Between
them a hook grows without a jump: with no hook the cut is across the long
leg; over the first sample's spacing of hook (a quarter of the tip) it
turns with the last two samples to lie across the hook, as it did, and
the rounded outside of the corner comes in behind it; from there to half
a tip the cut crosses less and less of the band before the bend, and at
half a tip it is clear of it. A tip 24 wide drawn 140 pixels and turned
square for 2, 4 and 6 pixels has, 4 to 12 pixels before the corner, 0.92,
1.09 and 1.29 times as much on the inside half of the stroke as on the
outside (it is the inside of a bend; 1.19, 1.29 and 1.34 with samples an
eighth of the tip apart), and in the band before the bend
exactly what the same stroke has with round ends, pixel for pixel, up to
the last sample before the corner; so has every hook of an eighth of the
tip to just under half of it, for tips 6 to 48 wide, at the end of a
stroke or at its start and turning either way. (The cut took whatever lay
past the line of all that the tip laid within reach: of the band before
such a hook it left as little as 0.81, and at the worst pixel nothing; the
inside half was 0.66, 0.53 and 0.87 of the outside half for those three,
and as little as 0.34.)

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
   light one. What a pass lays shrinks as the room runs out. A bed of dust
   under a dry tip (dust beyond the crumbs of the tip's own mark) takes
   none of it: the tip ploughs through it or rides on it, and lays a
   little less for the dust that rolls under it.
4. **Deposit.** What one pass lays depends on the medium, the pressure and
   the paper's bite, not on how much the tooth holds; so a toothier paper
   takes more passes to fill. It is split between `held` and `loose` by the
   medium's `binding`, and it is what the tip loses. Of the share that is
   held, a waxy medium lays as a film on top whatever met dust and not the
   paper; a dry one, what met a bed of dust.
5. **Working what is there.** The tip presses some of the dust under it into
   `held`. At the end of the pass its own dust (or paste) and its film
   settle on top of what was there, which goes `under`: all but a little of
   it once what is new is thick enough to cover it. A film of wax and
   paste bond part of the loose dust they come to lie on; the film of a
   dry medium binds nothing.
6. **Wax.** A waxy medium films over the tooth, which then takes less
   (resist). How much less depends on what is laid over it: a dry medium,
   which the tooth has to scrape off, hardly takes at all; a waxy one takes a
   little; a greasy paste such as oil pastel sticks to it. It also depends
   on how thick the film is, and not alike for both: wax goes on taking wax,
   less and less, until the film is `WAX_FULL` thick, so that light layers of
   coloured pencil build; but a dry medium has no tooth left to scrape it
   off as soon as the wax covers the tooth, which is the thickness at which
   any layer covers what is under it (`COVERS`). One firm line of crayon is
   that thick, so charcoal shaded over it skips it: wax resist. Pressed hard, the tip polishes the film: it raises `polish`
   towards the level its pressure reaches (see "Pressed flat"),
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
   at 0.6, the polish smoothed over the grain spread by 0.78 of its mean,
   and then by 0.58; of an oil pastel 1.08, then 0.73; 0.41 and 0.57 now
   that each stroke polishes to the level of its pressure.) How much a tip polishes
   goes with how slick the film is, and it takes up `BURNISH_LIFT`
   times its polish of what the paper holds. A colourless blender is made
   for this: soft clear wax and nothing else, it melts into the film
   (`Medium.melts`), so it polishes a thin film as if it were slick, takes
   up `MELT_LIFT` times its polish and lays it back within about its own
   width: one firm layer of it stirs about half of what is under it into an
   even film. On three layers of red at 0.5 on cartridge paper,
   a tenth of the pixels still show the paper; one layer of the blender at
   full pressure leaves a sixth of those (two layers a thirtieth), the
   chroma of the patch goes from 34 to 61 (70), its lightness from 67 to 57
   (54), it is half as grainy, and it has 0.45 (0.76) of the gloss of
   polished wax. A pale coloured pencil is mostly wax and white filler, and
   melts the more the paler it is (`pencils.melts_of`: a white one 0.75 as
   far as the blender, a cream 0.54, a pale blue 0.31, a pink or a mid grey
   hardly, a strong colour not at all; assumed, not measured): one firm
   layer of white over the red leaves a fifth of the flecks (two fifths
   before, when it melted no more than the red), 0.24 of gloss (0.17), and
   a paler tint of the red as light as it was (L* 65.5, chroma 41, where
   the blender takes it to 57 and 61). At a pressure of 0.7 the blender closes two fifths of the
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
   and raises `crush`, towards a level that goes with how hard it is: the
   hardest lead's is `FLATTEN` at full
   pressure (more than flat: it crushes the tooth flat from far less), a
   softer one's less as the cube of its hardness (it gives before the
   paper does), and that of one that crumbles to dust as the square of its binding
   (it bears on the paper only with what holds together). Three firm layers
   leave the tooth flattened by 0.89 under
   a 4H, 0.34 under an HB, 0.16 under a 2B and 0.013 under a 6B, and six
   layers by little more (0.95, 0.37, 0.18 and 0.014: their levels are 1,
   0.38, 0.18 and 0.014); compressed
   charcoal and hard pastel flatten it by a few hundredths, vine charcoal and
   soft pastel next to not at all. (While every pass added to what the
   passes before had crushed, three layers of HB made 0.74 and six 0.99: any
   lead flattened the tooth as a hard one does, given layers enough.) A flattened tooth is touched all over, and bites
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
paper, nearly two thirds lies within one width of the edge. What a tool
lifts into its load is the small share that goes far, and a stump or a
finger lifts little (`Tool.lift_loose`: an eighth of the dust in a pass for
a stump, most of which goes deep into it, and a thirtieth for a finger):
the trail of that stump is a fortieth of the tone three widths out and goes
on, fainter.

The load is what a tool tones with, and three rules make a stump that has
been used for long a different thing from one dragged once through a tone.
A stump or a finger holds so much on its surface (`Tool.holds`, a volume
for each pixel of its face) and lifts the less the fuller that is: it fills
slowly, over a hundred widths and more of rubbing in a tone, and once full
it takes nothing more. It gives the dust of its load to bare tooth only
(`hunger`): where the paper carries nothing it lays `Tool.lay` of its load
over each of its widths, less where pigment lies already, and nothing once
that covers the tooth (`COVERS` thick, on the covered share of the pixel).
And what it lifts works into it: a stump is paper rolled tight, and as it
is rubbed what is on its surface works into its fibres, which keep a deep
load of their own (`Tool.deep`, a volume for each pixel of its face, more
than its surface holds), and works back out to the surface as bare paper
empties that. The two even out towards being as full as each other, the gap
between them closing by `1 - exp(-SOAK_IN)` for each width of the tool
rubbed at full pressure as the surface gives to the deep load, and by
`1 - exp(-SOAK_OUT)`, about a quarter as fast, as it takes from it
(`soak`); what moves is a share of all the tool carries, pigment, paste and
wax alike, and the deep load is part of `Smudger.load`. A finger keeps a
little in the grain of its skin (`FINGER.deep` is three tenths of what it
holds); an eraser, a chamois and a tool that holds whatever it takes keep
none.

So a full stump leaves a tone as it is, however often it goes over it, and
a finger loaded with one colour leaves a tone of another clean, where an
exchange both ways would bring its colour all over it; but on bare paper
the same stump draws a tone of its own. The surface is what it lays from,
so it governs how much the stump carries out of a tone, and it stays
shallow: one pass through 32 widths of vine charcoal leaves 0.14 to 0.17 of
what it holds there (it was 0.11 to 0.13 before the stump had a deep load),
while the deep load takes nearly as much again: the stump has lifted 0.26 to
0.32 in all (it was 0.11 to 0.13), and on bare paper it lays 0.97 and 0.98
of the paper's tone 16 and 24 widths on (it was 0.98 and 0.99), still
carrying half of what it lifted. Loaded on a patch of 6B, it lays 0.745 of
the paper's tone where it lands, 0.85 eight of its widths on, 0.89 after
sixteen and 0.92 after twenty-four, by when it has laid half of what it
carried (with no deep load: 0.73, 0.84, 0.90 and 0.945, and seven eighths
of it; in the first build 0.59, 0.92, 0.98 and 0.99). Rubbed for long in
that patch it is full through and through, and drawn over 28 widths of bare
paper again and again it lands at 0.70, 0.90, 0.93, 0.96 and 0.97 of the
paper's tone, where with no deep load it was spent in two. The price is
what a clean stump takes off a tone: it fills its deep load too, so a bare
stump (clean at every rub) takes about 10% of a dark tone of 8B in eight
rubs, where it took 5% with only its surface to fill, and a stump in the
hand takes 1.4% of it before it is full, where it took 0.6%. A full stump
also lays dust on whatever bare paper it strays onto at the edge of a tone,
as a dirty stump does. Paste sticks to any paper, bare or
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
a layer of those `COVERS` thick). The film a dry medium leaves on a bed of
dust has no wax: a tool does not slide on it, and takes it with the loose
dust it lies on (see "A bed of dust").

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
and a dab is one pass of the tool on the paper under it. (It used to rub
only as it slid on: at the ends of its path an eraser took two thirds of
what it took further on, and a dab a quarter.)

The hand that rubs an area (`Sheet.rub`, `hatching.passes`) is not the hand
that shades one. A pencil's strokes are flicked on and off the paper: their
ends stray half a width either way, past the edge as often as short of it,
and are lighter where the pencil lands and lifts. Someone rubbing a tone
with a stump or a finger works the tool up to the edge, where they can see
it, and turns there, bearing as they go. So the middle of the tool keeps
half its width inside the outline of the area, and its edge comes to the
area's edge and no further: every pass runs from that line to that line,
the outermost passes run along it, and the others lie evenly between. In
the middle of the area the hand is as unsteady as a shading's (the passes
stray across and wander), but it keeps each pass within `SPREAD`, 0.6 of
the tool's width, of the one before: passes half a width apart that each
strayed by 0.15 of a width either way were now and then a width apart, and
left a band between them that only the rims of the tool reached, rubbed a
fifth as much as the rest (a row across the passes was rubbed below a
quarter as much as the middle in one rub in twenty); where the tool meets
the edge it is guided by it,
as steady as anywhere along a pass: the ends of the passes and the passes
along the edge stray only as a point of a pass wanders (`WANDER`, a tenth
of the tool's width, where the ends of a pencil's strokes stray by
`STRAY_ALONG`, half of it), and only inwards, and the tool does not
lighten where it turns. The points it goes through, and the curve through
them, are kept half a width inside the outline, of a polygon too, at any
angle; a tool wider than the area goes down its middle. Since a loaded tool
gives its dust to bare tooth, the old hand dirtied the paper round every
rubbed tone: rubbing the patch of charcoal in the third row of
`examples/crayons.png` with a clean finger, which loads as it rubs, darkened
the bare paper beside it by up to 105 levels of 255 (its luminance from
0.87 to 0.25), on 2132 pixels, as far as 15 pixels past the ends of the
passes and 7 past their sides; it leaves it as it was. And the ends of the
passes, which strayed and lightened, are rubbed as their middles are: an
eraser pressed lightly takes as much a pixel in from the end of its passes
as in the middle of the area (0.94, was 0.54; three pixels in 1.01, was
0.64), a stump smooths the grain as much two pixels in (0.98, was 0.76).
What a tool kept inside cannot do is rub the last pixels along the sides
of the outermost passes as well as the rest: only its rounded rim reaches
them. Along those sides the outermost pixel is rubbed a tenth to a third
as much as the middle and the area is rubbed as evenly as its middle from
about a third of the tool's width in, where the old hand, whose tool
reached a quarter of its width past the edge there, rubbed the outermost
pixel 0.7 to 0.9 as much as the middle; a second rub across the first
works them up to the edge. A caller
who wants the tool to go past the edge, as it used to, gives a larger area.
What a tool carries is in three parts: what it has lifted, which it lays
back at its own rate; the dust on its face, which it lays back within a few
pixels (in each step it shares out among the pixels under it what was on
its face as the step began, each by how it bears there); and the held
graphite it is pushing along, likewise. All three stay on a tool
in the hand (a `Smudger`) when it is lifted, and it lands with them; what a
bare tool still carries at the end is counted and returned. Either way
nothing is lost.

A rub takes the same numbers from the sheet's generator whatever its
pressure: the places of its passes (which go by the area, the tool, the angle
and the hand) and one seed, from which the seed of each stretch it rubs is
derived, as an integer step (`_seed_of`) from the sample of the rub that the
stretch starts at. Where the pressure is nothing the tool is lifted and the
passes fall into stretches, and a pass can fall into none; were each stretch
to draw its own seed, a rub that followed what was measured of the drawing
(a field a few thousandths off) would give another number of draws, and the
grain of every mark after it would move. One seed to the rub, not to the
pass: the number of passes goes by the area, the tool and the hand and so
does not depend on the pressure, but a rub that skipped a pass for what it
measures would, with a seed to each pass, shift the rest. A stretch's seed is
the same wherever the pressure is nothing elsewhere.

A shading, likewise, takes one number from the sheet's generator however
many strokes it lays. A stick's width follows its wear, which follows the
pressure of all it drew before, and the width sets the spacing and so how
many strokes a shading takes: were the hand and each stroke's seed drawn
from the sheet's generator, a stick worn a little more or less would move
the grain of every mark after the shading. The number seeds a generator of
the shading's own, from which the hand is drawn, and each stroke's seed is
derived from it and the stroke's place in the shading (`_seed_of`), so that
it does not depend on how many numbers the hand took before it.

A mark can also be given a seed of its own (`seed=`, an integer from 0 to
2**31 - 1, to `stroke`, `shade`, `smudge`, `erase` and `rub`). The grain of a
mark is salted with an integer that the sheet draws from its generator, so a
sheet that has drawn more or fewer marks before gives the next one another
grain, and a drawing in which one part changes (a few pixels hidden, a stroke
more) changes in every mark laid after it. With a seed, the integer a stroke,
a smudge or an erase gives its kernel is the seed itself, a `shade` takes it
in place of the one number it would have drawn, and a `rub`, which also draws
the places and the unsteadiness of its passes, draws those and its one seed
from `default_rng(seed)`:
none of them touches the sheet's generator, so the grain of a seeded mark is
the same whatever was drawn before it and what is drawn after it is as if it
had not been drawn. A caller that wants a drawing to stay as it is where it
is not changed keys the seed to the mark (its part, its place, its number).
Without a seed everything is as it was, to the bit.

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
in, so a crayon drawn over charcoal dust is seen over it, and so is a
pencil line drawn over a bed of pastel.
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
  fifths as thick shines as much: three firm layers of 4H shine 2.4 times as
  much as the same graphite would on a tooth that had kept its shape (0.31
  against 0.13 of the most), and of HB, which crush it by a third, 1.25
  times; six of HB shine three fifths as unevenly. A soft lead shines by how much it lays, a hard one by how flat
  it has pressed the paper. The light is white, in a broad lobe, since
  graphite is satin and not a mirror. Under the default lamp, three firm
  layers of 6B reflect about 0.10 of what the paper does, against 0.06
  without the shine: graphite never looks as black as charcoal.
  A pencil drawn over a bed of dust lays the graphite the paper would have
  held as a film on the dust (see "A bed of dust"), and that film shines as
  what the paper holds does. A layer is one mixture: what its film is made
  of, and what lies over what within it, are not kept. What can be told is
  that the graphite in a layer with a film came with a pencil's mark, the
  film and its crumbs; so the film is taken to hold the share of it that
  the paper holds of a pencil's mark (`FILM_GRAPHITE`, its binding), never
  more than its own volume, which is exact for a pencil over pastel, and
  loose graphite dust from elsewhere that lies with a film is taken for its
  crumbs, matte but for the film's share. In the layer on top, the film
  shines through all the matte that lies with it: the clay of its crumbs,
  as a line on paper does through its own, and whatever dust is there, for
  dust a light pass did not bury cannot be told from dust laid over the
  film after it. That errs on the matte side, as loose graphite is matte: a
  light line shines less than its film would. A film that a later pass
  buried lies on the older dust it was buried in, and shines through the
  clay of its crumbs and the matte of the layer on top. Where a pixel has
  wax, a film there is taken for one of wax, as a tool takes it. On
  cartridge paper, along lines 4 wide drawn over two layers of a pale soft
  pastel, of which 0.88 to 0.90 of the graphite is a film where on bare
  paper the paper holds it: a 2B at full pressure shines 0.38 (0.43 on bare
  paper; nothing before, 0.001), a 6B 0.68 (0.76), an HB 0.27 (0.31), the
  same over the pastel fixed; at 0.6 a 2B 0.18 (0.22), at 0.3 a third of
  what it shines on bare paper. Overhead, the lamp adds 0.022 of the
  paper's white to the 2B line (0.025 on bare paper) and 0.040 to the 6B
  (0.044). Six layers of 6B at full pressure shaded over the pastel shine
  0.78, against 0.82 on bare paper. A stump or a finger drawn across a line
  on loose pastel takes its film with the pastel and its shine with it; on
  fixed pastel, 0.83 of the film stays and 0.87 and 0.81 of its shine;
  and what they spread is dust, with no film, and shines as it did.
- **Wax** has no sheen of its own: it is its polish that shines. The gloss is
  how slick the film of wax is at the pixel (as for the resist) times the
  square of how far the wax is polished (`polish`; a film half polished still
  scatters most of the light), so a coloured pencil laid lightly has
  none, one burnished with a blender three quarters of the most, and two
  layers of crayon from none at a pressure of 0.3 to seven tenths at full
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
  L* 7 to 10 under it and to 18 overhead, and the gloss that the firm
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
  and no rubber eraser touches it: an eraser takes the pencil sketch and
  leaves the ink. A pencil draws over it as on bare paper. What cuts the
  surface of the paper takes the ink stained into it: a knife or a sand
  eraser, by the share it scrapes of what it lifts of what the paper holds
  (`scrape_ink`), and the ink is then on the tool, counted with what it
  took. Two layers of firm black hatching (a fifth of the paper's tone): a
  knife drawn across them at 0.9 scratches a line back to 0.98 of the
  paper's tone, leaving half a hundredth of the ink, and one rub of a sand
  eraser takes nine tenths of it, to 0.79. (No tool touched the ink.)
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
  line beyond it). A blob is a round dot a little wider than the line
  (`BLOB_WIDE` times the disc of the ball that meets the paper there, which
  is all of it pressed firmly and narrower pressed lightly, as the line
  is: on cartridge paper a pen six pixels wide leaves a blob of radius 3.1
  at a pressure of 0.05 and 4.2 from 0.5 up, where it was 4.2 at any
  pressure), thickest at its middle, and it is ink the pen lost like the
  rest: `stroke` returns it with the line. A drop lies in the valleys of
  the tooth as on its peaks, a firm pass thick at least, wherever the line
  laid its ink: it inks as much of a pixel's bare paper as it would cover
  of the pixel so, and a tiny drop next to nothing. (It inked all of a
  pixel that held a firm pass of ink, however small: a billionth of a firm
  pass, left every four pixels along a heavy line, darkened it by a
  seventh.) How much a pen gathers is its own (`Pen.blots`: 0 for one that
  never blots, more than 1 for a leaky one), apart from how freely its ink
  flows (`Pen.flow`, which darkens the line too).
- **A pen put to the paper has ink on its ball already.** A pen in the
  hand (`HeldPen`) lands with what its last line gathered and did not
  leave: of that, where it lands, a share drawn at random comes round,
  mostly next to none, so that one line in four or so starts with a dot
  that would show on a firm line, and a few with a blob; the rest comes
  round at the first sharp turn. It keeps the ink on its ball until it is
  wiped; what it gathered is on the paper or on its ball. Its ball holds a
  drop at most (`BALL_BLOB`), and what it would gather beyond that is not
  gathered: it goes nowhere, neither onto the paper nor onto the ball. So
  the identity (the ink of its drops and the ink on its ball are
  `BALL_GATHER` times that of its lines) holds only while the rim has never
  been full; long lines with no sharp turn, a hatching's, fill it (a pen
  three pixels wide: a straight line 130 pixels long at full pressure, 240
  at 0.5, 360 at 0.3) and keep it full. Where that ink would go (a smear
  along the line, onto the hand) is not modelled. A bare `Pen` is picked
  up afresh for every line: it lands with as much as such a share of the
  most its ball holds, which it leaves where it lands, and takes away what
  it gathers.
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
