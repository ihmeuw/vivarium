from vivarium.framework.resource import Resource


def test_resource_id() -> None:
    resource = Resource("value_source", "test")
    assert resource.resource_id == "value_source.test"
