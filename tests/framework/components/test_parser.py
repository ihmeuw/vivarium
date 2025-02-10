from abc import ABCMeta
from typing import Any
from unittest.mock import Mock, call

import pytest
import yaml
from _pytest.monkeypatch import MonkeyPatch
from layered_config_tree.main import LayeredConfigTree
from pytest_mock import MockerFixture

from tests.helpers import MockComponentA, MockComponentB
from vivarium.framework.components.parser import ComponentConfigurationParser, ParsingError
from vivarium.framework.configuration import build_simulation_configuration

TEST_COMPONENTS_NESTED = """
components:
    ministry:
        silly_walk:
           - Prance()
           - Jump('front_flip')
           - PratFall('15')
    pet_shop:
       - Parrot()
       - dangerous_animals.Crocodile('gold_tooth', 'teacup', '3.14')
    news: SomethingCompletelyDifferent()
"""

# common case when users comment out and ends up having none
TEST_COMPONENTS_EMPTY_KEY = """
components:
    ministry.silly_walk:
       - Prance()
       - Jump('front_flip')
       - PratFall('15')
    pet_shop:
#       - Parrot()
#       - dangerous_animals.Crocodile('gold_tooth', 'teacup', '3.14')
    news: SomethingCompletelyDifferent()
"""

TEST_COMPONENTS_NON_STRING = """
components:
    ministry.silly_walk:
       - 42
       - Jump('front_flip')
       - PratFall('15')
    pet_shop:
#       - Parrot()
#       - dangerous_animals.Crocodile('gold_tooth', 'teacup', '3.14')
    news: SomethingCompletelyDifferent()
"""

TEST_COMPONENTS_FLAT = """
components:
    - ministry.silly_walk.Prance()
    - ministry.silly_walk.Jump('front_flip')
    - ministry.silly_walk.PratFall('15')
    - pet_shop.Parrot()
    - pet_shop.dangerous_animals.Crocodile('gold_tooth', 'teacup', '3.14')
    - news.SomethingCompletelyDifferent()
"""

TEST_COMPONENTS_PARSED = [
    "ministry.silly_walk.Prance()",
    "ministry.silly_walk.Jump('front_flip')",
    "ministry.silly_walk.PratFall('15')",
    "pet_shop.Parrot()",
    "pet_shop.dangerous_animals.Crocodile('gold_tooth', 'teacup', '3.14')",
    "news.SomethingCompletelyDifferent()",
]

TEST_COMPONENTS_PREPPED = [
    ("ministry.silly_walk.Prance", tuple()),
    ("ministry.silly_walk.Jump", ("front_flip",)),
    ("ministry.silly_walk.PratFall", ("15",)),
    ("pet_shop.Parrot", tuple()),
    ("pet_shop.dangerous_animals.Crocodile", ("gold_tooth", "teacup", "3.14")),
    ("news.SomethingCompletelyDifferent", tuple()),
]


def mock_importer(path: str) -> ABCMeta:
    return {
        "test_components.MockComponentA": MockComponentA,
        "test_components.MockComponentB": MockComponentB,
    }[path]


@pytest.fixture()
def parser(mocker: MockerFixture) -> ComponentConfigurationParser:
    parser = ComponentConfigurationParser()
    parser.import_and_instantiate_component = mocker.Mock()  # type: ignore [method-assign]
    return parser


@pytest.fixture(params=[TEST_COMPONENTS_NESTED, TEST_COMPONENTS_FLAT])
def components(request: pytest.FixtureRequest) -> str:
    assert isinstance(request.param, str)
    return request.param


def test_prep_component(parser: ComponentConfigurationParser) -> None:
    desc = 'cave_system.monsters.Rabbit("timid", "squeak")'
    component, args = parser.prep_component(desc)
    assert component == "cave_system.monsters.Rabbit"
    assert set(args) == {"timid", "squeak"}


def test_prep_component_syntax_error(parser: ComponentConfigurationParser) -> None:
    desc = 'cave_system.monsters.Rabbit("timid", 0.01)'
    with pytest.raises(ParsingError):
        parser.prep_component(desc)

    desc = 'cave_system.monsters.Rabbit("timid\', "0.01")'
    with pytest.raises(ParsingError):
        parser.prep_component(desc)

    desc = "cave_system.monsters.Rabbit(\"timid', '0.01')"
    with pytest.raises(ParsingError):
        parser.prep_component(desc)


def test_parse_and_prep_components(parser: ComponentConfigurationParser) -> None:
    prepped_components = [
        parser.prep_component(component) for component in TEST_COMPONENTS_PARSED
    ]

    assert set(TEST_COMPONENTS_PREPPED) == set(prepped_components)


def test_import_and_instantiate_components(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("vivarium.framework.components.parser.import_by_path", mock_importer)

    component_descriptions = [
        ("test_components.MockComponentA", ("A Hundred and One Ways to Start a Fight",)),
        ("test_components.MockComponentB", ("Ethel the Aardvark goes Quantity Surveying",)),
    ]

    parser = ComponentConfigurationParser()
    component_list = [
        parser.import_and_instantiate_component(path, arg_tuple)
        for path, arg_tuple in component_descriptions
    ]

    assert len(component_list) == 2
    assert isinstance(component_list[0], MockComponentA)
    assert component_list[0].args == ("A Hundred and One Ways to Start a Fight",)
    assert isinstance(component_list[1], MockComponentB)
    assert component_list[1].args == ("Ethel the Aardvark goes Quantity Surveying",)


def test_get_components(parser: ComponentConfigurationParser, components: str) -> None:
    config = build_simulation_configuration()
    config.update(components)

    parser.get_components(config.components)

    calls = [call(path, args) for path, args in TEST_COMPONENTS_PREPPED]
    # mypy is interpreting assert_has_calls as an attribute
    parser.import_and_instantiate_component.assert_has_calls(calls)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "config, error_message",
    [
        (TEST_COMPONENTS_EMPTY_KEY, "empty with the header"),
        (TEST_COMPONENTS_NON_STRING, "should be a string, list, or dictionary"),
    ],
)
def test_components_invalid_config(
    parser: ComponentConfigurationParser, config: str, error_message: str
) -> None:
    bad_config = LayeredConfigTree(yaml.full_load(config))["components"]
    with pytest.raises(ParsingError, match=error_message):
        parser.parse_component_config(bad_config)
