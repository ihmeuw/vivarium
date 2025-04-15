from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from layered_config_tree import LayeredConfigTree
from vivarium_testing_utils import FuzzyChecker

from vivarium import InteractiveContext
from vivarium.framework.utilities import from_yearly


def test_disease_model(fuzzy_checker: FuzzyChecker, disease_model_spec: Path) -> None:
    config = LayeredConfigTree(disease_model_spec, layers=["base", "override"])
    config.update(
        {
            "configuration": {
                "mortality": {
                    "mortality_rate": 20.0,
                },
                "lower_respiratory_infections": {
                    "incidence_rate": 25.0,
                    "remission_rate": 50.0,
                    "excess_mortality_rate": 0.01,
                },
            },
        }
    )

    simulation = InteractiveContext(config)

    pop = simulation.get_population()
    expected_columns = {
        "tracked",
        "alive",
        "age",
        "sex",
        "entrance_time",
        "lower_respiratory_infections",
        "child_wasting_propensity",
    }
    assert set(pop.columns) == expected_columns
    assert len(pop) == 100_000
    assert np.all(pop["tracked"] == True)
    assert np.all(pop["alive"] == "alive")
    assert np.all((pop["age"] >= 0) & (pop["age"] <= 5))
    assert np.all(pop["entrance_time"] == datetime(2021, 12, 31, 12))

    for sex in ["Female", "Male"]:
        fuzzy_checker.fuzzy_assert_proportion(
            observed_numerator=(pop["sex"] == sex).sum(),
            observed_denominator=len(pop),
            target_proportion=0.5,
            # todo: remove this parameter when MIC-5412 is resolved
            name=f"{sex}_proportion",
        )

    assert np.all(pop["lower_respiratory_infections"] == "susceptible_to_lower_respiratory_infections")
    assert np.all((pop["child_wasting_propensity"] >= 0) & (pop["child_wasting_propensity"] <= 1))

    simulation.step()
    pop = simulation.get_population()
    is_alive = pop["alive"] == "alive"

    fuzzy_checker.fuzzy_assert_proportion(
        observed_numerator=(len(pop[~is_alive])),
        observed_denominator=len(pop),
        target_proportion=from_yearly(20, timedelta(days=0.5)),
        # todo: remove this parameter when MIC-5412 is resolved
        name="alive_proportion",
    )

    has_lri = pop["lower_respiratory_infections"] == "infected_with_lower_respiratory_infections"
    fuzzy_checker.fuzzy_assert_proportion(
        observed_numerator=(len(pop[is_alive & has_lri])),
        observed_denominator=len(pop[is_alive]),
        target_proportion=from_yearly(25, timedelta(days=0.5)),
        # todo: remove this parameter when MIC-5412 is resolved
        name="lri_proportion",
    )

    # todo test remission and excess mortality
    # todo test risk factor and intervention
