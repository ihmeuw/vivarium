import pandas as pd
import pytest

from vivarium.framework.results import ResultsInterface, ResultsManager


def _silly_aggregator(_: pd.DataFrame) -> float:
    return 1.0


@pytest.mark.parametrize(
    (
        "name, pop_filter, aggregator, requires_columns, requires_values,"
        " additional_stratifications, excluded_stratifications, when"
    ),
    [
        (
            "living_person_time",
            'alive == "alive" and undead == False',
            _silly_aggregator,
            None,
            None,
            [],
            [],
            "collect_metrics",
        ),
        (
            "undead_person_time",
            "undead == True",
            _silly_aggregator,
            None,
            None,
            [],
            [],
            "time_step__prepare",
        ),
    ],
    ids=["valid_on_collect_metrics", "valid_on_time_step__prepare"],
)
def test_register_observation(
    name,
    pop_filter,
    aggregator,
    requires_columns,
    requires_values,
    additional_stratifications,
    excluded_stratifications,
    when,
):
    mgr = ResultsManager()
    interface = ResultsInterface(mgr)
    # interface.set_default_stratifications(["age", "sex"])
    assert len(interface._manager._results_context._observations) == 0
    interface.register_observation(
        name,
        pop_filter,
        aggregator,
        additional_stratifications,
        excluded_stratifications,
    )
    assert len(interface._manager._results_context._observations) == 1
