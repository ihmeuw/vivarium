from tests.helpers import ColumnCreator
from vivarium.framework.resource import Resource


def test_resource_id() -> None:
    resource = Resource("test", ColumnCreator())
    assert resource.resource_id == "generic_resource.test"
