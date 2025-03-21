import pytest
from layered_config_tree.main import LayeredConfigTree
from pytest_mock import MockerFixture

from vivarium.framework.engine import Builder
from vivarium.framework.results.observer import Observer


class TestObserver(Observer):
    def register_observations(self, builder: Builder) -> None:
        pass


class TestDefaultObserverStratifications(Observer):
    def register_observations(self, builder: Builder) -> None:
        pass


class TestObserverStratifications(Observer):
    def register_observations(self, builder: Builder) -> None:
        pass

    @property
    def configuration_defaults(self) -> dict[str, str]:
        return {"foo": "bar"}


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
