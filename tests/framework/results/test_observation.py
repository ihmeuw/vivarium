from __future__ import annotations

import itertools
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from tests.framework.results.helpers import BASE_POPULATION, FAMILIARS, HOUSE_CATEGORIES
from vivarium.framework.results import VALUE_COLUMN
from vivarium.framework.results.context import ResultsContext
from vivarium.framework.results.observation import (
    AddingObservation,
    ConcatenatingObservation,
    StratifiedObservation,
)


@pytest.fixture
def stratified_observation() -> StratifiedObservation:
    return StratifiedObservation(
        name="stratified_observation_name",
        pop_filter="",
        when="whenevs",
        results_updater=lambda _, __: pd.DataFrame(),
        results_formatter=lambda _, __: pd.DataFrame(),
        stratifications=(),
        aggregator_sources=None,
        aggregator=lambda _: 0.0,
    )


@pytest.fixture
def concatenating_observation() -> ConcatenatingObservation:
    return ConcatenatingObservation(
        name="concatenating_observation_name",
        pop_filter="",
        when="whenevs",
        included_columns=["some-col", "some-other-col"],
        results_formatter=lambda _, __: pd.DataFrame(),
    )


@pytest.mark.parametrize(
    "stratifications, aggregator_sources, aggregator",
    [
        # Series or single-column dataframe return
        (("familiar",), ["power_level"], len),
        (("familiar",), [], len),
        (("familiar", "house"), ["power_level"], len),
        (("familiar", "house"), [], len),
        ((), ["power_level"], len),
        ((), [], len),
        # Multiple-column dataframe return
        (("familiar",), ["power_level", "tracked"], sum),
        (("familiar", "house"), ["power_level", "tracked"], sum),
        ((), ["power_level", "tracked"], sum),
    ],
)
def test_stratified_observation__aggregate(
    stratifications: tuple[str, ...],
    aggregator_sources: list[str],
    aggregator: Callable[[pd.DataFrame], float | pd.Series[float]],
    stratified_observation: StratifiedObservation,
) -> None:
    """Test that we are aggregating correctly. There are some nuances here:
    - If aggregator_resources is provided, then simply .apply it to the groups passed in.
    - If no aggregator_resources are provided, then we want a full aggregation of the groups.
    - _aggregate can return either a pd.Series or a pd.DataFrame of any number of columns
    """

    filtered_pop = BASE_POPULATION.copy()
    for stratification in stratifications:
        mapped_col = f"{stratification}_mapped_values"
        filtered_pop[mapped_col] = filtered_pop[stratification]
    groups = ResultsContext()._get_groups(
        stratifications=stratifications, filtered_pop=filtered_pop
    )
    aggregates = stratified_observation._aggregate(
        pop_groups=groups,  # type: ignore [arg-type]
        aggregator_sources=aggregator_sources,
        aggregator=aggregator,
    )
    if aggregator == len:
        if stratifications:
            stratification_idx: set[tuple[str, ...] | str] = (
                set(itertools.product(*(FAMILIARS, HOUSE_CATEGORIES)))
                if "house" in stratifications
                else set(FAMILIARS)
            )
            assert set(aggregates.index) == stratification_idx
            check = pd.Series(aggregates.values == len(BASE_POPULATION) / groups.ngroups)
            assert check.all()
        else:
            assert len(aggregates.values) == 1
            assert aggregates.values[0] == len(BASE_POPULATION)
    else:  # sum aggregator
        assert aggregates.shape[1] == 2
        expected = BASE_POPULATION[["power_level", "tracked"]].sum() / groups.ngroups
        if stratifications:
            stratification_idx = (
                set(itertools.product(*(FAMILIARS, HOUSE_CATEGORIES)))
                if "house" in stratifications
                else set(FAMILIARS)
            )
            assert set(aggregates.index) == stratification_idx
            final = aggregates.sum() / groups.ngroups
            assert isinstance(final, pd.Series)
            assert final.equals(expected)
        else:
            assert len(aggregates.values) == 1
            for col in ["power_level", "tracked"]:
                assert aggregates.loc["all", col] == expected[col]


@pytest.mark.parametrize(
    "aggregates",
    [
        pd.Series(data=[1, 2, 3], index=pd.Index(["a", "b", "c"], name="index")),
        pd.DataFrame({"col1": [1, 2], "strat1": [1, 1], "strat2": ["cat", "dog"]}).set_index(
            ["strat1", "strat2"]
        ),
    ],
)
def test_stratified_observation__format(
    aggregates: pd.DataFrame | pd.Series[float], stratified_observation: StratifiedObservation
) -> None:
    new_aggregates = stratified_observation._format(aggregates=aggregates)
    assert isinstance(new_aggregates, pd.DataFrame)
    if isinstance(aggregates, pd.Series):
        assert new_aggregates.equals(aggregates.to_frame("value"))
    else:
        assert new_aggregates.equals(aggregates)


@pytest.mark.parametrize(
    "aggregates",
    [
        pd.DataFrame(
            {VALUE_COLUMN: [1.0, 2.0, 10.0, 20.0]},
            index=pd.Index(["ones"] * 2 + ["tens"] * 2),
        ),
        pd.DataFrame(
            {VALUE_COLUMN: [1.0, 2.0, 10.0, 20.0, "bad", "bad"]},
            index=pd.MultiIndex.from_arrays(
                [
                    ["foo", "bar", "foo", "bar", "foo", "bar"],
                    ["ones", "ones", "tens", "tens", "zeros", "zeros"],
                ],
                names=["nonsense", "type"],
            ),
        ).query('type!="zeros"'),
    ],
)
def test_stratified_observation__expand_index(
    aggregates: pd.DataFrame, stratified_observation: StratifiedObservation
) -> None:
    full_idx_aggregates = stratified_observation._expand_index(aggregates=aggregates)
    # NOTE: pd.MultiIndex is a subclass of pd.Index, i.e. check for this first!
    if isinstance(aggregates.index, pd.MultiIndex):
        # Check that index is cartesian product of the original index levels
        assert full_idx_aggregates.index.equals(
            pd.MultiIndex.from_product(aggregates.index.levels)
        )
        # Check that existing values did not change
        assert (
            full_idx_aggregates.loc[aggregates.index, VALUE_COLUMN]
            == aggregates[VALUE_COLUMN]
        ).all()
        # Check that missingness was filled in with zeros
        assert (full_idx_aggregates.query('type=="zeros"')[VALUE_COLUMN] == 0).all()
    else:
        assert aggregates.equals(full_idx_aggregates)


@pytest.mark.parametrize(
    "stratifications",
    [
        ("familiar",),
        ("familiar", "house"),
        (),
    ],
)
def test_stratified_observation_results_gatherer(
    stratifications: tuple[str, ...], stratified_observation: StratifiedObservation
) -> None:
    ctx = ResultsContext()
    # Append the post-stratified columns
    filtered_population = BASE_POPULATION.copy()
    for stratification in stratifications:
        mapped_col = f"{stratification}_mapped_values"
        filtered_population[mapped_col] = filtered_population[stratification]
    pop_groups = ctx._get_groups(
        stratifications=stratifications, filtered_pop=filtered_population
    )
    df = stratified_observation.results_gatherer(pop_groups, stratifications)
    ctx._rename_stratification_columns(df)
    assert set(df.columns) == set(["value"])
    expected_idx_names = (
        list(stratifications) if len(stratifications) > 0 else ["stratification"]
    )
    assert list(df.index.names) == expected_idx_names


@pytest.mark.parametrize(
    "new_observations",
    [
        pd.DataFrame({"value": [1.0, 2.0]}),
        pd.DataFrame(
            {
                "value": [1.0, 2.0],
                "another_value": [3.0, 4.0],
                "yet_another_value": [5.0, 6.0],
            }
        ),
        pd.DataFrame({"another_value": [3.0, 4.0], "yet_another_value": [5.0, 6.0]}),
    ],
)
def test_adding_observation_results_updater(new_observations: pd.DataFrame) -> None:
    existing_results = pd.DataFrame({"value": [0.0, 0.0]})
    obs = AddingObservation(
        name="adding_observation_name",
        pop_filter="",
        when="whenevs",
        results_formatter=lambda _, __: pd.DataFrame(),
        stratifications=(),
        aggregator_sources=None,
        aggregator=lambda _: 0.0,
    )
    updated_results = obs.results_updater(existing_results, new_observations)
    if "value" in new_observations.columns:
        assert updated_results.equals(new_observations)
    else:
        assert updated_results.equals(pd.concat([existing_results, new_observations], axis=1))


@pytest.mark.parametrize(
    "new_observations, expected_results",
    [
        (
            pd.DataFrame({"value": ["two", "three"]}),
            pd.DataFrame({"value": ["zero", "one", "two", "three"]}),
        ),
        (
            pd.DataFrame(
                {
                    "another_value": ["foo", "bar"],
                    "yet_another_value": ["cat", "dog"],
                }
            ),
            pd.DataFrame(
                {
                    "value": ["zero", "one", np.nan, np.nan],
                    "another_value": [np.nan, np.nan, "foo", "bar"],
                    "yet_another_value": [np.nan, np.nan, "cat", "dog"],
                }
            ),
        ),
    ],
)
def test_concatenating_observation_results_updater(
    new_observations: pd.DataFrame,
    expected_results: pd.DataFrame,
    concatenating_observation: ConcatenatingObservation,
) -> None:
    existing_results = pd.DataFrame({"value": ["zero", "one"]})
    updated_results = concatenating_observation.results_updater(
        existing_results, new_observations
    )
    assert updated_results.equals(expected_results)
