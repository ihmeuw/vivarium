import itertools
import math
import re
from collections.abc import Callable
from datetime import timedelta
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from layered_config_tree.main import LayeredConfigTree
from loguru import logger
from pandas.core.groupby.generic import DataFrameGroupBy
from pytest_mock import MockerFixture

from tests.framework.results.helpers import (
    BASE_POPULATION,
    FAMILIARS,
    HOUSE_CATEGORIES,
    NAME,
    NAME_COLUMNS,
    sorting_hat_serial,
    sorting_hat_vectorized,
    verify_stratification_added,
)
from vivarium.framework.event import Event
from vivarium.framework.lifecycle import lifecycle_states
from vivarium.framework.results import VALUE_COLUMN
from vivarium.framework.results.context import ResultsContext
from vivarium.framework.results.observation import AddingObservation, ConcatenatingObservation
from vivarium.framework.results.stratification import Stratification, get_mapped_col_name
from vivarium.framework.values import Pipeline
from vivarium.types import ScalarMapper, VectorMapper


def _aggregate_state_person_time(x: pd.DataFrame) -> float:
    """Helper aggregator function for observation testing"""
    return len(x) * (28 / 365.25)


@pytest.fixture
def event() -> Event:
    return Event(
        name=lifecycle_states.COLLECT_METRICS,
        index=pd.Index([0]),
        user_data={},
        time=0,
        step_size=1,
    )


@pytest.mark.parametrize(
    "mapper, is_vectorized",
    [
        (sorting_hat_vectorized, True),
        (sorting_hat_serial, False),
    ],
    ids=["vectorized_mapper", "non-vectorized_mapper"],
)
def test_add_stratification_mappers(
    mapper: VectorMapper | ScalarMapper, is_vectorized: bool, mocker: MockerFixture
) -> None:
    ctx = ResultsContext()
    mocker.patch.object(ctx, "excluded_categories", {})
    pipeline = Pipeline("grade")

    assert NAME not in ctx.stratifications

    ctx.add_stratification(
        name=NAME,
        requires_columns=NAME_COLUMNS,
        requires_values=[pipeline],
        categories=HOUSE_CATEGORIES,
        excluded_categories=None,
        mapper=mapper,
        is_vectorized=is_vectorized,
    )
    assert verify_stratification_added(
        stratifications=ctx.stratifications,
        name=NAME,
        requires_columns=NAME_COLUMNS,
        requires_values=[pipeline],
        categories=HOUSE_CATEGORIES,
        excluded_categories=[],
        mapper=mapper,
        is_vectorized=is_vectorized,
    )


@pytest.mark.parametrize(
    "excluded_categories",
    [
        [],
        HOUSE_CATEGORIES[:1],
        HOUSE_CATEGORIES[:2],
        HOUSE_CATEGORIES[:3],
    ],
    ids=[
        "no_excluded_categories",
        "one_excluded_category",
        "two_excluded_categories",
        "all_but_one_excluded_categories",
    ],
)
def test_add_stratification_excluded_categories(
    excluded_categories: list[str], mocker: MockerFixture
) -> None:
    ctx = ResultsContext()
    builder = mocker.Mock()
    builder.configuration.stratification = LayeredConfigTree(
        {"default": [], "excluded_categories": {NAME: excluded_categories}}
    )
    builder.logging.get_logger.return_value = logger
    ctx.setup(builder)

    assert NAME not in ctx.stratifications

    ctx.add_stratification(
        name=NAME,
        requires_columns=NAME_COLUMNS,
        requires_values=[],
        categories=HOUSE_CATEGORIES,
        excluded_categories=excluded_categories,
        mapper=sorting_hat_vectorized,
        is_vectorized=True,
    )

    assert verify_stratification_added(
        stratifications=ctx.stratifications,
        name=NAME,
        requires_columns=NAME_COLUMNS,
        requires_values=[],
        categories=HOUSE_CATEGORIES,
        excluded_categories=excluded_categories,
        mapper=sorting_hat_vectorized,
        is_vectorized=True,
    )


@pytest.mark.parametrize(
    "name, categories, excluded_categories, msg_match",
    [
        (
            "duplicate_name",
            HOUSE_CATEGORIES,
            [],
            "Stratification name 'duplicate_name' is already used",
        ),
        (
            NAME,
            HOUSE_CATEGORIES + ["slytherin"],
            [],
            f"Found duplicate categories in stratification '{NAME}': ['slytherin']",
        ),
        (
            NAME,
            HOUSE_CATEGORIES + ["gryffindor", "slytherin"],
            [],
            f"Found duplicate categories in stratification '{NAME}': ['gryffindor', 'slytherin']",
        ),
        (
            NAME,
            HOUSE_CATEGORIES,
            ["gryfflepuff"],
            "Excluded categories {'gryfflepuff'} not found in categories",
        ),
    ],
    ids=[
        "duplicate_name",
        "duplicate_category",
        "duplicate_categories",
        "unknown_excluded_category",
    ],
)
def test_add_stratification_raises(
    name: str,
    categories: list[str],
    excluded_categories: list[str],
    msg_match: str,
    mocker: MockerFixture,
) -> None:
    ctx = ResultsContext()
    mocker.patch.object(ctx, "excluded_categories", {name: excluded_categories})
    # Register a stratification to test against duplicate stratifications
    ctx.add_stratification(
        name="duplicate_name",
        requires_columns=["foo"],
        requires_values=[],
        categories=["bar"],
        excluded_categories=None,
        mapper=sorting_hat_serial,
        is_vectorized=False,
    )
    with pytest.raises(ValueError, match=re.escape(msg_match)):
        ctx.add_stratification(
            name=name,
            requires_columns=NAME_COLUMNS,
            requires_values=[],
            categories=categories,
            excluded_categories=excluded_categories,
            mapper=sorting_hat_vectorized,
            is_vectorized=True,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "name": "living_person_time",
            "pop_filter": 'alive == "alive" and undead == False',
            "requires_columns": ["alive", "undead"],
            "when": lifecycle_states.COLLECT_METRICS,
        },
        {
            "name": "undead_person_time",
            "pop_filter": "undead == True",
            "requires_columns": ["undead"],
            "when": lifecycle_states.TIME_STEP_PREPARE,
        },
    ],
    ids=["valid_on_collect_metrics", "valid_on_time_step__prepare"],
)
def test_register_observation(kwargs: Any) -> None:
    ctx = ResultsContext()
    assert len(ctx.grouped_observations) == 0
    kwargs["results_formatter"] = lambda: None
    kwargs["stratifications"] = tuple()
    kwargs["aggregator_sources"] = []
    kwargs["aggregator"] = len
    kwargs["requires_values"] = []
    ctx.register_observation(
        observation_type=AddingObservation,
        **kwargs,
    )
    assert len(ctx.grouped_observations) == 1


def test_register_observation_duplicate_name_raises() -> None:
    ctx = ResultsContext()
    ctx.register_observation(
        observation_type=AddingObservation,
        name="some-observation-name",
        pop_filter="some-pop-filter",
        when="some-when",
        requires_columns=[],
        requires_values=[],
        results_formatter=lambda df: df,
        stratifications=(),
        aggregator_sources=[],
        aggregator=len,
    )
    with pytest.raises(
        ValueError, match="Observation name 'some-observation-name' is already used: "
    ):
        # register a different observation but w/ the same name
        ctx.register_observation(
            observation_type=ConcatenatingObservation,
            name="some-observation-name",
            pop_filter="some-other-pop-filter",
            when="some-other-when",
            requires_columns=[],
            requires_values=[],
            stratifications=None,
        )


@pytest.mark.parametrize(
    "aggregator_sources, aggregator, stratifications",
    [
        ([], len, ["house", "familiar"]),
        (["power_level"], sum, ["house", "familiar"]),
        ([], _aggregate_state_person_time, ["house", "familiar"]),
        ([], len, ["house"]),
        (["power_level"], sum, ["house"]),
        ([], _aggregate_state_person_time, ["house"]),
    ],
    ids=[
        "len_aggregator_two_stratifications",
        "sum_aggregator_two_stratifications",
        "custom_aggregator_two_stratifications",
        "len_aggregator_one_stratification",
        "sum_aggregator_one_stratification",
        "custom_aggregator_one_stratification",
    ],
)
def test_adding_observation_gather_results(
    aggregator_sources: list[str],
    aggregator: Callable[..., int | float],
    stratifications: list[str],
    event: Event,
) -> None:
    """Test cases where every stratification is in gather_results. Checks for
    existence and correctness of results"""
    ctx = ResultsContext()

    # Generate population DataFrame
    population = BASE_POPULATION.copy()
    for stratification in stratifications:
        population[get_mapped_col_name(stratification)] = population[stratification]

    # Set up stratifications
    if "house" in stratifications:
        ctx.add_stratification(
            name="house",
            requires_columns=["house"],
            requires_values=[],
            categories=HOUSE_CATEGORIES,
            excluded_categories=None,
            mapper=None,
            is_vectorized=True,
        )
    if "familiar" in stratifications:
        ctx.add_stratification(
            name="familiar",
            requires_columns=["familiar"],
            requires_values=[],
            categories=FAMILIARS,
            excluded_categories=None,
            mapper=None,
            is_vectorized=True,
        )
    pop_filter = "tracked==True"
    observation = ctx.register_observation(
        observation_type=AddingObservation,
        name="foo",
        pop_filter=pop_filter,
        requires_columns=aggregator_sources,
        requires_values=[],
        aggregator_sources=aggregator_sources,
        aggregator=aggregator,
        stratifications=tuple(stratifications),
        when=lifecycle_states.COLLECT_METRICS,
        results_formatter=lambda: None,
    )

    filtered_pop = population.query(pop_filter)
    groups = filtered_pop.groupby(stratifications)
    if aggregator == sum:
        power_level_sums = groups[aggregator_sources].sum().squeeze()
        assert len(power_level_sums.unique()) == 1
        expected_result = power_level_sums.iat[0]
    else:
        group_sizes = groups.size()
        assert len(group_sizes.unique()) == 1
        num_stratifications = group_sizes.iat[0]
        expected_result = (
            num_stratifications if aggregator == len else num_stratifications * 28 / 365.25
        )

    i = 0
    for result, _measure, _updater in ctx.gather_results(
        population, event.name, [observation]
    ):
        assert result is not None
        assert all(
            math.isclose(actual_result, expected_result, rel_tol=0.0001)
            for actual_result in result.values
        )
        i += 1
    assert i == 1


def test_concatenating_observation_gather_results(event: Event) -> None:

    ctx = ResultsContext()

    # Generate population DataFrame
    population = BASE_POPULATION.copy()
    # Mock out some extra columns that would be produced by the manager's _prepare_population() method
    population["current_time"] = pd.Timestamp(year=2045, month=1, day=1, hour=12)
    population["event_step_size"] = timedelta(days=28)
    population["event_time"] = pd.Timestamp(year=2045, month=1, day=1, hour=12) + timedelta(
        days=28
    )

    lifecycle_state = lifecycle_states.COLLECT_METRICS
    pop_filter = "house=='hufflepuff'"
    included_cols = ["familiar", "house"]
    observation = ctx.register_observation(
        observation_type=ConcatenatingObservation,
        name="foo",
        pop_filter=pop_filter,
        when=lifecycle_state,
        requires_columns=included_cols,
        requires_values=[],
        results_formatter=lambda _, __: pd.DataFrame(),
        stratifications=None,
    )

    filtered_pop = population.query(pop_filter)

    i = 0
    for result, _measure, _updater in ctx.gather_results(
        population, event.name, [observation]
    ):
        assert result is not None
        assert result.equals(filtered_pop[["event_time"] + included_cols])
        i += 1
    assert i == 1


@pytest.mark.parametrize(
    "name, aggregator_sources, aggregator, stratifications",
    [
        ("wizard_count", [], len, ["house", "familiar"]),
        ("power_level_total", ["power_level"], sum, ["house", "familiar"]),
        (
            "wizard_time",
            [],
            _aggregate_state_person_time,
            ["house", "familiar"],
        ),
        ("wizard_count", [], len, ["familiar"]),
        ("power_level_total", ["power_level"], sum, ["familiar"]),
        (
            "wizard_time",
            [],
            _aggregate_state_person_time,
            ["familiar"],
        ),
    ],
    ids=[
        "len_aggregator_two_stratifications",
        "sum_aggregator_two_stratifications",
        "custom_aggregator_two_stratifications",
        "len_aggregator_one_stratification",
        "sum_aggregator_one_stratification",
        "custom_aggregator_one_stratification",
    ],
)
def test_gather_results_partial_stratifications_in_results(
    name: str,
    aggregator_sources: list[str],
    aggregator: Callable[..., int | float],
    stratifications: list[str],
    event: Event,
) -> None:
    """Test cases where not all stratifications are observed for gather_results. This looks for existence of
    unobserved stratifications and ensures their values are 0"""
    ctx = ResultsContext()

    # Generate population DataFrame
    population = BASE_POPULATION.copy()
    # Remove an entire category from a stratification
    population = population[population["familiar"] != "unladen_swallow"].reset_index()

    # Set up stratifications
    if "house" in stratifications:
        ctx.add_stratification(
            name="house",
            requires_columns=["house"],
            requires_values=[],
            categories=HOUSE_CATEGORIES,
            excluded_categories=None,
            mapper=None,
            is_vectorized=True,
        )
        population[get_mapped_col_name("house")] = population["house"].astype(
            pd.CategoricalDtype(categories=HOUSE_CATEGORIES, ordered=True)
        )
    if "familiar" in stratifications:
        ctx.add_stratification(
            name="familiar",
            requires_columns=["familiar"],
            requires_values=[],
            categories=FAMILIARS,
            excluded_categories=None,
            mapper=None,
            is_vectorized=True,
        )
        population[get_mapped_col_name("familiar")] = population["familiar"].astype(
            pd.CategoricalDtype(categories=FAMILIARS, ordered=True)
        )

    observation = ctx.register_observation(
        observation_type=AddingObservation,
        name=name,
        pop_filter="tracked==True",
        requires_columns=aggregator_sources,
        requires_values=[],
        aggregator_sources=aggregator_sources,
        aggregator=aggregator,
        stratifications=tuple(stratifications),
        when=lifecycle_states.COLLECT_METRICS,
        results_formatter=lambda: None,
    )

    for results, _measure, _formatter in ctx.gather_results(
        population, event.name, [observation]
    ):
        assert results is not None
        unladen_results = results.reset_index().query('familiar=="unladen_swallow"')
        assert len(unladen_results) > 0
        assert (unladen_results[VALUE_COLUMN] == 0).all()


def test_gather_results_with_empty_pop_filter(event: Event) -> None:
    """Test case where pop_filter filters to an empty population. gather_results
    should return None.
    """
    ctx = ResultsContext()

    # Generate population DataFrame
    population = BASE_POPULATION.copy()

    lifecycle_state = lifecycle_states.COLLECT_METRICS
    observation = ctx.register_observation(
        observation_type=AddingObservation,
        name="wizard_count",
        pop_filter="house == 'durmstrang'",
        requires_columns=["house"],
        requires_values=[],
        aggregator_sources=[],
        aggregator=len,
        stratifications=tuple(),
        when=lifecycle_state,
        results_formatter=lambda: None,
    )

    for result, _measure, _updater in ctx.gather_results(
        population, event.name, [observation]
    ):
        assert not result


def test_gather_results_with_no_stratifications(event: Event) -> None:
    """Test case where we have no stratifications. gather_results should return one value."""
    ctx = ResultsContext()

    # Generate population DataFrame
    population = BASE_POPULATION.copy()

    lifecycle_state = lifecycle_states.COLLECT_METRICS
    observation = ctx.register_observation(
        observation_type=AddingObservation,
        name="wizard_count",
        pop_filter="",
        requires_columns=[],
        requires_values=[],
        aggregator_sources=None,
        aggregator=len,
        stratifications=tuple(),
        when=lifecycle_state,
        results_formatter=lambda: None,
    )

    assert len(ctx.stratifications) == 0
    assert (
        len(
            list(
                result
                for result, _measure, _updater in ctx.gather_results(
                    population, event.name, [observation]
                )
            )
        )
        == 1
    )


def test_bad_aggregator_stratification(event: Event) -> None:
    """Test if an exception gets raised when a stratification that doesn't
    exist is attempted to be used, as expected."""
    ctx = ResultsContext()

    # Generate population DataFrame
    population = BASE_POPULATION.copy()
    lifecycle_state = lifecycle_states.COLLECT_METRICS

    # Set up stratifications
    ctx.add_stratification(
        name="house",
        requires_columns=["house"],
        requires_values=[],
        categories=HOUSE_CATEGORIES,
        excluded_categories=None,
        mapper=None,
        is_vectorized=True,
    )
    ctx.add_stratification(
        name="familiar",
        requires_columns=["familiar"],
        requires_values=[],
        categories=FAMILIARS,
        excluded_categories=None,
        mapper=None,
        is_vectorized=True,
    )
    observation = ctx.register_observation(
        observation_type=AddingObservation,
        name="this_shouldnt_work",
        pop_filter="",
        requires_columns=[],
        requires_values=[],
        aggregator_sources=[],
        aggregator=sum,
        stratifications=("house", "height"),  # `height` is not a stratification
        when=lifecycle_state,
        results_formatter=lambda: None,
    )

    with pytest.raises(KeyError, match="height"):
        for result, _measure, _updater in ctx.gather_results(
            population, event.name, [observation]
        ):
            print(result)


@pytest.mark.parametrize(
    ["lifecycle_state", "time", "expected_observations"],
    [
        (lifecycle_states.COLLECT_METRICS, 0, ["obs1", "obs3"]),
        (lifecycle_states.TIME_STEP_PREPARE, 0, ["obs2"]),
        (lifecycle_states.COLLECT_METRICS, 1, ["obs1"]),
    ],
    ids=["collect_metrics_time_0", "time_step_prepare_time_0", "collect_metrics_time_1"],
)
def test_get_observations(
    lifecycle_state: str, time: int, expected_observations: list[str]
) -> None:
    ctx = ResultsContext()
    register_observation_kwargs = {
        "observation_type": AddingObservation,
        "pop_filter": "",
        "requires_columns": [],
        "requires_values": [],
        "results_formatter": lambda: None,
        "stratifications": (),
        "aggregator_sources": None,
        "aggregator": len,
    }

    ctx.register_observation(
        name="obs1", when=lifecycle_states.COLLECT_METRICS, **register_observation_kwargs  # type: ignore[arg-type]
    )
    ctx.register_observation(
        name="obs2", when=lifecycle_states.TIME_STEP_PREPARE, **register_observation_kwargs  # type: ignore[arg-type]
    )
    ctx.register_observation(
        name="obs3",
        when=lifecycle_states.COLLECT_METRICS,
        to_observe=lambda event: event.time == 0,
        **register_observation_kwargs,  # type: ignore[arg-type]
    )

    event = Event(
        name=lifecycle_state, index=pd.Index([0]), user_data={}, time=time, step_size=1
    )

    assert [obs.name for obs in ctx.get_observations(event)] == expected_observations


@pytest.mark.parametrize("resource_type", ["columns", "values"])
@pytest.mark.parametrize(
    "observation_names, stratification_names, expected_resources",
    [
        (["obs1", "obs2"], ["strat1", "strat2"], {"x", "y", "z", "v"}),
        (["obs3"], ["strat1", "strat2"], {"x", "y", "w", "v"}),
        ([], ["strat1"], {"x", "y"}),
        (["obs2"], [], {"y", "z"}),
        ([], [], set()),
    ],
    ids=[
        "obs_and_strat_with_overlap",
        "obs_and_strat_without_overlap",
        "no_observations",
        "no_stratifications",
        "neither",
    ],
)
def test_get_required_resources(
    resource_type: str,
    observation_names: list[str],
    stratification_names: list[str],
    expected_resources: set[str],
) -> None:
    ctx = ResultsContext()

    all_observations = {}
    register_observation_kwargs = {
        "observation_type": AddingObservation,
        "pop_filter": "",
        "when": lifecycle_states.COLLECT_METRICS,
        "results_formatter": lambda: None,
        "stratifications": (),
        "aggregator_sources": None,
        "aggregator": len,
    }

    def get_required_resources_kwargs(
        resource_type: str, resources: list[str]
    ) -> dict[str, list[str] | list[Pipeline]]:
        if resource_type == "columns":
            return {"requires_columns": resources, "requires_values": []}
        elif resource_type == "values":
            return {
                "requires_values": [Pipeline(r) for r in resources],
                "requires_columns": [],
            }
        else:
            raise ValueError(f"Unknown resource_type: {resource_type}")

    all_observations["obs1"] = ctx.register_observation(
        name="obs1",
        **get_required_resources_kwargs(resource_type, ["x", "y"]),  # type: ignore[arg-type]
        **register_observation_kwargs,  # type: ignore[arg-type]
    )
    all_observations["obs2"] = ctx.register_observation(
        name="obs2",
        **get_required_resources_kwargs(resource_type, ["y", "z"]),  # type: ignore[arg-type]
        **register_observation_kwargs,  # type: ignore[arg-type]
    )
    all_observations["obs3"] = ctx.register_observation(
        name="obs3",
        **get_required_resources_kwargs(resource_type, ["w"]),  # type: ignore[arg-type]
        **register_observation_kwargs,  # type: ignore[arg-type]
    )

    all_stratifications = {}
    stratification_kwargs = {
        "categories": ["cat1", "cat2"],
        "excluded_categories": ["cat3"],
        "mapper": lambda df: df["a"] + df["b"],
        "is_vectorized": True,
    }
    all_stratifications["strat1"] = Stratification(
        name="strat1",
        **get_required_resources_kwargs(resource_type, ["x", "y"]),  # type: ignore[arg-type]
        **stratification_kwargs,  # type: ignore[arg-type]
    )
    all_stratifications["strat2"] = Stratification(
        name="strat2",
        **get_required_resources_kwargs(resource_type, ["x", "v"]),  # type: ignore[arg-type]
        **stratification_kwargs,  # type: ignore[arg-type]
    )

    observations = [all_observations[name] for name in observation_names]
    stratifications = [all_stratifications[name] for name in stratification_names]

    if resource_type == "columns":
        actual_columns = ctx.get_required_columns(observations, stratifications)
        assert set(actual_columns) == {"tracked"} | expected_resources
    elif resource_type == "values":
        actual_columns = [
            p.name for p in ctx.get_required_values(observations, stratifications)
        ]
        assert set(actual_columns) == expected_resources
    else:
        raise ValueError(f"Unknown resource_type: {resource_type}")


@pytest.mark.parametrize(
    "pop_filter",
    ['familiar=="cat"', 'familiar=="spaghetti_yeti"', ""],
    ids=["pop_filter", "pop_filter_empties_dataframe", "no_pop_filter"],
)
def test__filter_population(pop_filter: str) -> None:
    population = BASE_POPULATION.copy()

    filtered_pop = ResultsContext()._filter_population(
        population=population, pop_filter=pop_filter
    )
    expected = population.copy()
    if pop_filter:
        familiar = pop_filter.split("==")[1].strip('"')
        expected = expected[expected["familiar"] == familiar]
    assert filtered_pop.equals(expected)


@pytest.mark.parametrize(
    "stratifications",
    [tuple(), ("new_col1",), ("new_col1", "new_col2")],
    ids=[
        "no_stratifications",
        "single_excluded_stratification",
        "two_excluded_stratifications",
    ],
)
def test__drop_na_stratifications(stratifications: tuple[str, ...]) -> None:
    population = BASE_POPULATION.copy()
    population["new_col1"] = "new_value1"
    population.loc[population["tracked"] == True, "new_col1"] = np.nan
    population["new_col2"] = "new_value2"
    population.loc[population["new_col1"].notna(), "new_col2"] = np.nan
    # Add on the post-stratified columns
    for stratification in stratifications:
        mapped_col = f"{stratification}_mapped_values"
        population[mapped_col] = population[stratification]

    filtered_pop = ResultsContext()._drop_na_stratifications(
        population=population, stratification_names=stratifications
    )
    expected = population.copy()
    for stratification in stratifications:
        expected = expected[expected[stratification].notna()]
    assert filtered_pop.equals(expected)


@pytest.mark.parametrize(
    "stratifications, values",
    [
        (("familiar",), [FAMILIARS]),
        (("familiar", "house"), [FAMILIARS, HOUSE_CATEGORIES]),
        ((), "foo"),
    ],
)
def test__get_groups(stratifications: tuple[str, ...], values: str | list[list[str]]) -> None:
    filtered_pop = BASE_POPULATION.copy()
    # Generate the post-stratified columns
    for stratification in stratifications:
        mapped_col = f"{stratification}_mapped_values"
        filtered_pop[mapped_col] = filtered_pop[stratification]
    groups = ResultsContext()._get_groups(
        stratifications=stratifications, filtered_pop=filtered_pop
    )
    assert isinstance(groups, DataFrameGroupBy)
    if stratifications:
        combinations = set(itertools.product(*values))
        if len(values) == 1:
            # convert from set of tuples to set of strings
            combinations = set(comb[0] for comb in combinations)
        # Check that all familiars exist
        assert set(groups.groups.keys()) == combinations
        # Check that the entire population is included
        assert sum([len(value) for value in groups.groups.values()]) == len(BASE_POPULATION)
    else:
        item = groups.groups.popitem()
        # Check that there are no other groups
        assert not groups.groups
        # Check that the group is 'all' and includes the entire population
        key, val = item
        assert key == "all"
        assert val.equals(BASE_POPULATION.index)
