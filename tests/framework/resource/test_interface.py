from pytest_mock import MockerFixture

from vivarium.framework.resource.interface import ResourceInterface
from vivarium.framework.resource.manager import ResourceManager
from vivarium.framework.resource.resource import Column


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
        dependencies=[],
    )
    resource_map = mgr._resource_group_map
    resource_ids = ["column.private_col_1", "column.private_col_2"]
    assert set(resource_map.keys()) == set(resource_ids)
    for resource_id in resource_ids:
        resource = resource_map[resource_id].resources[resource_id]
        assert isinstance(resource, Column)
        assert resource.name == resource_id.split(".", 1)[1]
        assert resource.resource_type == "column"
