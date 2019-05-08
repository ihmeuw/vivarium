import pytest
import yaml

from vivarium.framework.configuration import build_simulation_configuration
from vivarium.framework.components.parser import (ComponentConfigurationParser, parse_component_config_to_list,
                                                  prep_components, import_and_instantiate_components,
                                                  ParsingError)

from .mocks import MockComponentA, MockComponentB


TEST_COMPONENTS_NESTED = """
components:
    ministry.silly_walk:
       - Prance()
       - Jump('front_flip')
       - PratFall('15')
    pet_shop:
       - Parrot()
       - dangerous_animals.Crocodile('gold_tooth', 'teacup', '3.14')
"""

# common case when users comment out and ends up having none
TEST_COMPONENTS_BAD = """
components:
    ministry.silly_walk:
       - Prance()
       - Jump('front_flip')
       - PratFall('15')
    pet_shop:
#       - Parrot()
#       - dangerous_animals.Crocodile('gold_tooth', 'teacup', '3.14')
"""

TEST_COMPONENTS_FLAT = """
components:
    - ministry.silly_walk.Prance()
    - ministry.silly_walk.Jump('front_flip')
    - ministry.silly_walk.PratFall('15')
    - pet_shop.Parrot()
    - pet_shop.dangerous_animals.Crocodile('gold_tooth', 'teacup', '3.14')
"""

TEST_COMPONENTS_PARSED = ["ministry.silly_walk.Prance()",
                          "ministry.silly_walk.Jump('front_flip')",
                          "ministry.silly_walk.PratFall('15')",
                          "pet_shop.Parrot()",
                          "pet_shop.dangerous_animals.Crocodile('gold_tooth', 'teacup', '3.14')"]

TEST_COMPONENTS_PREPPED = [('ministry.silly_walk.Prance', tuple()),
                           ('ministry.silly_walk.Jump', ('front_flip',)),
                           ('ministry.silly_walk.PratFall', ('15',)),
                           ('pet_shop.Parrot', tuple()),
                           ('pet_shop.dangerous_animals.Crocodile', ('gold_tooth', 'teacup', '3.14'))]


def mock_importer(path):
    return {
        'test_components.MockComponentA': MockComponentA,
        'test_components.MockComponentB': MockComponentB,
    }[path]


@pytest.fixture(params=[TEST_COMPONENTS_NESTED, TEST_COMPONENTS_FLAT])
def components(request):
    return request.param


@pytest.fixture
def import_and_instantiate_mock(mocker):
    return mocker.patch('vivarium.framework.components.parser.import_and_instantiate_components')


def test_parse_component_config(components):

    source = yaml.full_load(components)['components']
    component_list = parse_component_config_to_list(source)
    assert set(TEST_COMPONENTS_PARSED) == set(component_list)


def test_prep_components():
    desc = 'cave_system.monsters.Rabbit("timid", "squeak")'
    component, args = prep_components([desc])[0]
    assert component == 'cave_system.monsters.Rabbit'
    assert set(args) == {'timid', 'squeak'}


def test_parse_component_syntax_error():
    desc = 'cave_system.monsters.Rabbit("timid", 0.01)'
    with pytest.raises(ParsingError):
        prep_components([desc])

    desc = 'cave_system.monsters.Rabbit("timid\', "0.01")'
    with pytest.raises(ParsingError):
        prep_components([desc])

    desc = "cave_system.monsters.Rabbit(\"timid', '0.01')"
    with pytest.raises(ParsingError):
        prep_components([desc])


def test_parse_and_prep_components(components):
    source = yaml.full_load(components)['components']
    prepped_components = prep_components(parse_component_config_to_list(source))

    assert set(TEST_COMPONENTS_PREPPED) == set(prepped_components)


def test_import_and_instantiate_components(monkeypatch):
    monkeypatch.setattr('vivarium.framework.components.parser.import_by_path', mock_importer)

    component_descriptions = [
        ('test_components.MockComponentA', ("A Hundred and One Ways to Start a Fight",)),
        ('test_components.MockComponentB', ("Ethel the Aardvark goes Quantity Surveying",)),
    ]
    component_list = import_and_instantiate_components(component_descriptions)

    assert len(component_list) == 2
    assert isinstance(component_list[0], MockComponentA)
    assert component_list[0].args == ("A Hundred and One Ways to Start a Fight",)
    assert isinstance(component_list[1], MockComponentB)
    assert component_list[1].args == ("Ethel the Aardvark goes Quantity Surveying",)


def test_ComponentConfigurationParser_get_components(import_and_instantiate_mock, components):
    config = build_simulation_configuration()
    config.update(components)

    parser = ComponentConfigurationParser()
    parser.get_components(config.components)

    import_and_instantiate_mock.assert_called_once_with(TEST_COMPONENTS_PREPPED)


def test_components_config_valid():
    bad_config = yaml.full_load(TEST_COMPONENTS_BAD)['components']
    with pytest.raises(ParsingError):
        parse_component_config_to_list(bad_config)
