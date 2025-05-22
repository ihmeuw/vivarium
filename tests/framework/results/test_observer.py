from typing import Any

import pytest
from layered_config_tree.main import LayeredConfigTree
from pytest_mock import MockerFixture

from vivarium import InteractiveContext
from vivarium.framework.components.manager import ComponentConfigError
from vivarium.framework.engine import Builder
from vivarium.framework.results.observer import Observer


class TestObserver(Observer):
    def register_observations(self, builder: Builder) -> None:
        pass


class TestDefaultObserverStratifications(Observer):
    def register_observations(self, builder: Builder) -> None:
        pass


class TestObserverStratifications(Observer):
    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            "stratification": {
                self.get_configuration_name(): {
                    "exclude": ["baz"],
                    "include": ["foo"],
                },
            },
        }

    def register_observations(self, builder: Builder) -> None:
        pass


def test_observer_instantiation() -> None:
    observer = TestObserver()
    assert observer.name == "test_observer"


@pytest.mark.parametrize(
    "is_interactive, results_dir",
    [
        (False, "/some/results/dir"),
        (True, None),
    ],
)
def test_set_results_dir(
    is_interactive: bool, results_dir: str | None, mocker: MockerFixture
) -> None:
    builder = mocker.Mock()
    if is_interactive:
        builder.configuration = LayeredConfigTree()
    else:
        builder.configuration = LayeredConfigTree(
            {
                "output_data": {"results_directory": results_dir},
            }
        )

    observer = TestObserver()
    observer.set_results_dir(builder)

    assert observer.results_dir == results_dir


def test_observer_get_configuration(
    base_config: LayeredConfigTree,
) -> None:

    observer = TestObserverStratifications()
    sim = InteractiveContext(
        base_config,
        components=[observer],
    )
    sim_observer_config = sim.configuration["stratification"][
        observer.get_configuration_name()
    ]
    # Observer.configuration calls get_configuration
    observer_config = observer.configuration
    assert observer_config is not None
    assert observer_config.to_dict() == dict(sim_observer_config)


def test_duplicated_observer_error(base_config: LayeredConfigTree) -> None:
    observer1 = TestObserverStratifications()
    observer2 = TestObserverStratifications()
    with pytest.raises(
        ComponentConfigError,
        match="is attempting to set the configuration value test, but it has already been set",
    ):
        InteractiveContext(
            base_config,
            components=[observer1, observer2],
        )
