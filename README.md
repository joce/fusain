# fusain

[![CI](https://github.com/joce/fusain/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/joce/fusain/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/joce/fusain/graph/badge.svg)](https://codecov.io/gh/joce/fusain)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub License](https://img.shields.io/github/license/joce/fusain)](https://github.com/joce/fusain/blob/main/LICENSE)

A drawing simulator for dry and semi-dry media: graphite and coloured pencils,
charcoal, pastel, crayon and the like, laid down on a paper with tooth. Wet
media (ink, felt-tip markers, paint) are out of scope; for oil paint, see the
sister project [oilpaint](https://github.com/joce/oilpaint). The two share their
colour science and stroke paths through [murex](https://github.com/joce/murex).

Early days. So far there is the paper, and a sheet of it that can be rendered;
nothing to draw with yet. The plan is in [docs/design.md](docs/design.md).

```python
from fusain import Sheet, papers

sheet = Sheet(800, 600, paper=papers.INGRES, seed=1)
sheet.save("paper.png", light=(-0.6, -0.8, 0.45))   # a raking light
```

![The papers under a raking light](examples/output/papers.png)

The papers of `fusain.papers` under a raking light: cartridge, printer paper,
Bristol, newsprint, Ingres, cold-pressed; rough, sanded card, toned grey, toned
tan, black.

## Development

```bash
uv sync --group dev --frozen   # install the package and the development tools
uv run pytest                  # run the tests
make pre-push                  # what CI runs: tox, coverage, spelling
```

## License

[MIT](LICENSE)
