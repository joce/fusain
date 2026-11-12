import fusain


def test_the_package_knows_its_version():
    assert isinstance(fusain.__version__, str)
    assert fusain.__version__


def test_everything_exported_exists():
    for name in fusain.__all__:
        assert getattr(fusain, name) is not None, name
