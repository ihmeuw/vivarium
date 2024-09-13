import pytest
from layered_config_tree.main import LayeredConfigTree

from vivarium.framework.results.observer import Observer


class TestObserver(Observer):
    def register_observations(self, builder):
        pass


class TestDefaultObserverStratifications(Observer):
    def register_observations(self, builder):
        pass


class TestObserverStratifications(Observer):
    def register_observations(self, builder):
        pass

    @property
    def configuration_defaults(self):
        return {"foo": "bar"}


def test_observer_instantiation():
    observer = TestObserver()
    assert observer.name == "test_observer"


@pytest.mark.parametrize(
    "is_interactive, results_dir",
    [
        (False, "/some/results/dir"),
        (True, None),
    ],
)
def test_set_results_dir(is_interactive, results_dir, mocker):
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


def test_diff_observer():
    from vivarium.framework.engine import SimulationContext
    sim = SimulationContext("/home/rmudambi/workspace/framework/vivarium/src/vivarium/examples/disease_model/disease_model.yaml")
    sim.run_simulation()
