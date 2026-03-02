from pytest_mock import MockerFixture

from vivarium.framework.resource.interface import ResourceInterface
from vivarium.framework.resource.manager import ResourceManager
from vivarium.framework.resource.resource import Column, ResourceId


def test_add_private_columns(mocker: MockerFixture) -> None:
    mgr = ResourceManager()
    interface = ResourceInterface(mgr)
    mocker.patch.object(
        mgr,
        "_get_current_component_or_manager",
        return_value=mocker.MagicMock(),
        create=True,
    )
    interface.add_private_columns(
        initializer=lambda pop_data: None,
        columns=["private_col_1", "private_col_2"],
        required_resources=[],
    )
    column_ids = [
        Column.get_resource_id("private_col_1"),
        Column.get_resource_id("private_col_2"),
    ]

    # Should have exactly the two columns and one initializer resource
    assert len(mgr._resources) == 3

    # Check that both columns exist
    for resource_id in column_ids:
        assert resource_id in mgr._resources
        resource = mgr._resources[resource_id]
        assert isinstance(resource, Column)
        assert resource.name == resource_id.name
        assert resource.RESOURCE_TYPE == "column"

    # Check that there's exactly one initializer resource
    initializer_resources = [
        r for r in mgr._resources.values() if r.RESOURCE_TYPE == "initializer"
    ]
    assert len(initializer_resources) == 1
