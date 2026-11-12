# fusain

[![CI](https://github.com/joce/fusain/actions/workflows/ci.yml/badge.svg)](https://github.com/joce/fusain/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/joce/fusain/graph/badge.svg)](https://codecov.io/gh/joce/fusain)

A drawing simulator for dry and semi-dry media: graphite and coloured pencils,
charcoal, pastel, crayon and the like, laid down on a paper with tooth. Wet
media (ink, felt-tip markers, paint) are out of scope; for oil paint, see the
sister project [oilpaint](https://github.com/joce/oilpaint).

Early days: nothing to draw with yet.

## Development

```bash
uv sync --group dev --frozen   # install the package and the development tools
uv run pytest                  # run the tests
make pre-push                  # what CI runs: tox, coverage, spelling
```

## License

MIT
