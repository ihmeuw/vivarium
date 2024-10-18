from vivarium.framework.resource import Resource


def test_to_string() -> None:
    resource = Resource("value_source", "test")
    assert str(resource) == "value_source.test"
