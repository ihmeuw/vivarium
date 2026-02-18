from tests.helpers import ColumnCreator
from vivarium.framework.resource import Resource


def test_resource_id() -> None:
    resource = Resource("value_source", "test", ColumnCreator())
    assert resource.resource_id == "value_source.test"
