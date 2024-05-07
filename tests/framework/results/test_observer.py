from pathlib import Path

import pandas as pd
import pytest
from layered_config_tree import LayeredConfigTree

from vivarium.framework.results.observer import Observer, StratifiedObserver


class TestObserver(Observer):
    def register_observations(self, builder):
        pass


class TestDefaultStratifiedObserver(StratifiedObserver):
    def register_observations(self, builder):
        pass


class TestStratifiedObserver(StratifiedObserver):
    def register_observations(self, builder):
        pass

    @property
    def configuration_defaults(self):
        return {"foo": "bar"}


def test_observer_instantiation():
    observer = TestObserver()
    assert observer.name == "test_observer"


@pytest.mark.parametrize(
    "is_interactive, results_dir, draw, seed",
    [
        (False, "/some/results/dir", 111, 222),
        (True, None, None, None),
    ],
)
def test_get_report_attributes(is_interactive, results_dir, draw, seed, mocker):
    builder = mocker.Mock()
    if is_interactive:
        builder.configuration = LayeredConfigTree()
    else:
        builder.configuration = LayeredConfigTree(
            {
                "output_data": {"results_directory": results_dir},
                "input_data": {"input_draw_number": draw},
                "randomness": {"random_seed": seed},
            }
        )

    observer = TestObserver()
    observer.get_report_attributes(builder)

    assert observer.results_dir == results_dir
    assert observer.input_draw == draw
    assert observer.random_seed == seed


def test_dataframe_to_csv(tmpdir, mocker):
    results_dir = Path(tmpdir)
    input_draw = 111
    random_seed = 222
    builder = mocker.Mock()
    builder.configuration = LayeredConfigTree(
        {
            "output_data": {"results_directory": results_dir},
            "input_data": {"input_draw_number": input_draw},
            "randomness": {"random_seed": random_seed},
        }
    )

    observer = TestObserver()
    observer.setup_component(builder)

    cats = pd.DataFrame(
        {
            "value": ["Whipper", "Burt Macklin"],
            "color": ["gray", "black"],
            "size": ["small", "large"],
        }
    ).set_index(["color", "size"])
    observer.dataframe_to_csv("cats", cats)

    expected_results = cats.reset_index()
    expected_results["input_draw"] = input_draw
    expected_results["random_seed"] = random_seed
    expected_results["measure"] = "cats"
    results = pd.read_csv(Path(tmpdir) / "cats.csv")

    assert set(results.columns) == set(expected_results.columns)

    # Order and sort for equality check
    cols = list(results.columns)
    results = results.sort_values(cols).reset_index(drop=True)
    expected_results = expected_results[cols].sort_values(cols).reset_index(drop=True)

    assert results.equals(expected_results)


@pytest.mark.parametrize(
    "observer, name, expected_configuration_defaults",
    [
        (
            TestDefaultStratifiedObserver(),
            "test_default_stratified_observer",
            {"stratification": {"test_default_stratified": {"exclude": [], "include": []}}},
        ),
        (TestStratifiedObserver(), "test_stratified_observer", {"foo": "bar"}),
    ],
)
def test_stratified_observer_instantiation(observer, name, expected_configuration_defaults):
    obs = observer
    assert obs.name == name
    assert obs.configuration_defaults == expected_configuration_defaults
