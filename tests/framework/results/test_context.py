import itertools
import math
import re
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from layered_config_tree.main import LayeredConfigTree
from loguru import logger
from pandas.core.groupby.generic import DataFrameGroupBy

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
from vivarium.framework.results import VALUE_COLUMN
from vivarium.framework.results.context import ResultsContext
from vivarium.framework.results.observation import AddingObservation, ConcatenatingObservation


def _aggregate_state_person_time(x: pd.DataFrame) -> float:
    """Helper aggregator function for observation testing"""
    return len(x) * (28 / 365.25)


@pytest.fixture
def mocked_event(mocker) -> Event:
    event: Event = mocker.Mock(spec=Event)
    return event


@pytest.mark.parametrize(
    "mapper, is_vectorized",
    [
        (sorting_hat_vectorized, True),
        (sorting_hat_serial, False),
    ],
    ids=["vectorized_mapper", "non-vectorized_mapper"],
)
def test_add_stratification_mappers(mapper, is_vectorized, mocker):
    ctx = ResultsContext()
    mocker.patch.object(ctx, "excluded_categories", {})
    assert not verify_stratification_added(
        ctx.stratifications, NAME, NAME_COLUMNS, HOUSE_CATEGORIES, [], mapper, is_vectorized
    )
    ctx.add_stratification(
        name=NAME,
        sources=NAME_COLUMNS,
        categories=HOUSE_CATEGORIES,
        excluded_categories=None,
        mapper=mapper,
        is_vectorized=is_vectorized,
    )
    assert verify_stratification_added(
        ctx.stratifications, NAME, NAME_COLUMNS, HOUSE_CATEGORIES, [], mapper, is_vectorized
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
def test_add_stratification_excluded_categories(excluded_categories, mocker):
    ctx = ResultsContext()
    builder = mocker.Mock()
    builder.configuration.stratification = LayeredConfigTree(
        {"default": [], "excluded_categories": {NAME: excluded_categories}}
    )
    builder.logging.get_logger.return_value = logger
    ctx.setup(builder)
    assert not verify_stratification_added(
        ctx.stratifications,
        NAME,
        NAME_COLUMNS,
        HOUSE_CATEGORIES,
        [],
        sorting_hat_vectorized,
        True,
    )
    ctx.add_stratification(
        name=NAME,
        sources=NAME_COLUMNS,
        categories=HOUSE_CATEGORIES,
        excluded_categories=excluded_categories,
        mapper=sorting_hat_vectorized,
        is_vectorized=True,
    )
    assert verify_stratification_added(
        ctx.stratifications,
        NAME,
        NAME_COLUMNS,
        HOUSE_CATEGORIES,
        excluded_categories,
        sorting_hat_vectorized,
        True,
    )


@pytest.mark.parametrize(
    "name, categories, excluded_categories, msg_match",
    [
        (
            "duplicate_name",
            HOUSE_CATEGORIES,
            [],
            "Stratification name 'duplicate_name' is already used: ",
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
def test_add_stratification_raises(name, categories, excluded_categories, msg_match, mocker):
    ctx = ResultsContext()
    mocker.patch.object(ctx, "excluded_categories", {name: excluded_categories})
    # Register a stratification to test against duplicate stratifications
    ctx.add_stratification(
        name="duplicate_name",
        sources=["foo"],
        categories=["bar"],
        excluded_categories=None,
        mapper=sorting_hat_serial,
        is_vectorized=False,
    )
    with pytest.raises(ValueError, match=re.escape(msg_match)):
        ctx.add_stratification(
            name=name,
            sources=NAME_COLUMNS,
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
            "when": "collect_metrics",
        },
        {
            "name": "undead_person_time",
            "pop_filter": "undead == True",
            "when": "time_step__prepare",
        },
    ],
    ids=["valid_on_collect_metrics", "valid_on_time_step__prepare"],
)
def test_register_observation(kwargs):
    ctx = ResultsContext()
    assert len(ctx.observations) == 0
    kwargs["results_formatter"] = lambda: None
    kwargs["stratifications"] = tuple()
    kwargs["aggregator_sources"] = []
    kwargs["aggregator"] = len
    ctx.register_observation(
        observation_type=AddingObservation,
        **kwargs,
    )
    assert len(ctx.observations) == 1


def test_register_observation_duplicate_name_raises():
    ctx = ResultsContext()
    ctx.register_observation(
        observation_type=AddingObservation,
        name="some-observation-name",
        pop_filter="some-pop-filter",
        when="some-when",
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
        )


@pytest.mark.parametrize(
    "pop_filter, aggregator_sources, aggregator, stratifications",
    [
        ("tracked==True", None, len, ["house", "familiar"]),
        (
            "tracked==True",
            ["power_level"],
            sum,
            ["house", "familiar"],
        ),
        (
            "tracked==True",
            [],
            _aggregate_state_person_time,
            ["house", "familiar"],
        ),
        ("tracked==True", None, len, ["house"]),
        ("tracked==True", ["power_level"], sum, ["house"]),
        (
            "tracked==True",
            [],
            _aggregate_state_person_time,
            ["house"],
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
def test_adding_observation_gather_results(
    pop_filter, aggregator_sources, aggregator, stratifications, mocked_event
):
    """Test cases where every stratification is in gather_results. Checks for
    existence and correctness of results"""
    ctx = ResultsContext()

    # Generate population DataFrame
    population = BASE_POPULATION.copy()
    # Mock out some extra columns that would be produced by the manager's _prepare_population() method
    population["current_time"] = pd.Timestamp(year=2045, month=1, day=1, hour=12)
    population["event_step_size"] = timedelta(days=28)
    population["event_time"] = pd.Timestamp(year=2045, month=1, day=1, hour=12) + timedelta(
        days=28
    )
    lifecycle_phase = "collect_metrics"

    # Set up stratifications
    if "house" in stratifications:
        ctx.add_stratification(
            name="house",
            sources=["house"],
            categories=HOUSE_CATEGORIES,
            excluded_categories=None,
            mapper=None,
            is_vectorized=True,
        )
    if "familiar" in stratifications:
        ctx.add_stratification(
            name="familiar",
            sources=["familiar"],
            categories=FAMILIARS,
            excluded_categories=None,
            mapper=None,
            is_vectorized=True,
        )
    ctx.register_observation(
        observation_type=AddingObservation,
        name="foo",
        pop_filter=pop_filter,
        aggregator_sources=aggregator_sources,
        aggregator=aggregator,
        stratifications=tuple(stratifications),
        when=lifecycle_phase,
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
        population, lifecycle_phase, mocked_event
    ):
        assert all(
            math.isclose(actual_result, expected_result, rel_tol=0.0001)
            for actual_result in result.values
        )
        i += 1
    assert i == 1


def test_concatenating_observation_gather_results(mocked_event):

    ctx = ResultsContext()

    # Generate population DataFrame
    population = BASE_POPULATION.copy()
    # Mock out some extra columns that would be produced by the manager's _prepare_population() method
    population["current_time"] = pd.Timestamp(year=2045, month=1, day=1, hour=12)
    population["event_step_size"] = timedelta(days=28)
    population["event_time"] = pd.Timestamp(year=2045, month=1, day=1, hour=12) + timedelta(
        days=28
    )

    lifecycle_phase = "collect_metrics"
    pop_filter = "house=='hufflepuff'"
    included_cols = ["event_time", "familiar", "house"]
    ctx.register_observation(
        observation_type=ConcatenatingObservation,
        name="foo",
        pop_filter=pop_filter,
        when=lifecycle_phase,
        included_columns=included_cols,
        results_formatter=lambda _, __: pd.DataFrame(),
    )

    filtered_pop = population.query(pop_filter)

    i = 0
    for result, _measure, _updater in ctx.gather_results(
        population, lifecycle_phase, mocked_event
    ):
        assert result.equals(filtered_pop[included_cols])
        i += 1
    assert i == 1


@pytest.mark.parametrize(
    "name, pop_filter, aggregator_sources, aggregator, stratifications",
    [
        ("wizard_count", "tracked==True", None, len, ["house", "familiar"]),
        ("power_level_total", "tracked==True", ["power_level"], sum, ["house", "familiar"]),
        (
            "wizard_time",
            "tracked==True",
            [],
            _aggregate_state_person_time,
            ["house", "familiar"],
        ),
        ("wizard_count", "tracked==True", None, len, ["familiar"]),
        ("power_level_total", "tracked==True", ["power_level"], sum, ["familiar"]),
        (
            "wizard_time",
            "tracked==True",
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
    name, pop_filter, aggregator_sources, aggregator, stratifications, mocked_event
):
    """Test cases where not all stratifications are observed for gather_results. This looks for existence of
    unobserved stratifications and ensures their values are 0"""
    ctx = ResultsContext()

    # Generate population DataFrame
    population = BASE_POPULATION.copy()

    # Mock out some extra columns that would be produced by the manager's _prepare_population() method
    population["current_time"] = pd.Timestamp(year=2045, month=1, day=1, hour=12)
    population["event_step_size"] = timedelta(days=28)
    population["event_time"] = pd.Timestamp(year=2045, month=1, day=1, hour=12) + timedelta(
        days=28
    )
    # Remove an entire category from a stratification
    population = population[population["familiar"] != "unladen_swallow"].reset_index()

    lifecycle_phase = "collect_metrics"

    # Set up stratifications
    if "house" in stratifications:
        ctx.add_stratification(
            name="house",
            sources=["house"],
            categories=HOUSE_CATEGORIES,
            excluded_categories=None,
            mapper=None,
            is_vectorized=True,
        )
    if "familiar" in stratifications:
        ctx.add_stratification(
            name="familiar",
            sources=["familiar"],
            categories=FAMILIARS,
            excluded_categories=None,
            mapper=None,
            is_vectorized=True,
        )

    ctx.register_observation(
        observation_type=AddingObservation,
        name=name,
        pop_filter=pop_filter,
        aggregator_sources=aggregator_sources,
        aggregator=aggregator,
        stratifications=tuple(stratifications),
        when=lifecycle_phase,
        results_formatter=lambda: None,
    )

    for results, _measure, _formatter in ctx.gather_results(
        population, lifecycle_phase, mocked_event
    ):
        unladen_results = results.reset_index().query('familiar=="unladen_swallow"')
        assert len(unladen_results) > 0
        assert (unladen_results[VALUE_COLUMN] == 0).all()


def test_gather_results_with_empty_pop_filter(mocked_event):
    """Test case where pop_filter filters to an empty population. gather_results
    should return None.
    """
    ctx = ResultsContext()

    # Generate population DataFrame
    population = BASE_POPULATION.copy()

    lifecycle_phase = "collect_metrics"
    ctx.register_observation(
        observation_type=AddingObservation,
        name="wizard_count",
        pop_filter="house == 'durmstrang'",
        aggregator_sources=[],
        aggregator=len,
        stratifications=tuple(),
        when=lifecycle_phase,
        results_formatter=lambda: None,
    )

    for result, _measure, _updater in ctx.gather_results(
        population, lifecycle_phase, mocked_event
    ):
        assert not result


def test_gather_results_with_no_stratifications(mocked_event):
    """Test case where we have no stratifications. gather_results should return one value."""
    ctx = ResultsContext()

    # Generate population DataFrame
    population = BASE_POPULATION.copy()

    lifecycle_phase = "collect_metrics"
    ctx.register_observation(
        observation_type=AddingObservation,
        name="wizard_count",
        pop_filter="",
        aggregator_sources=None,
        aggregator=len,
        stratifications=tuple(),
        when=lifecycle_phase,
        results_formatter=lambda: None,
    )

    assert len(ctx.stratifications) == 0
    assert (
        len(
            list(
                result
                for result, _measure, _updater in ctx.gather_results(
                    population, lifecycle_phase, mocked_event
                )
            )
        )
        == 1
    )


def test_bad_aggregator_stratification(mocked_event):
    """Test if an exception gets raised when a stratification that doesn't
    exist is attempted to be used, as expected."""
    ctx = ResultsContext()

    # Generate population DataFrame
    population = BASE_POPULATION.copy()
    lifecycle_phase = "collect_metrics"

    # Set up stratifications
    ctx.add_stratification(
        name="house",
        sources=["house"],
        categories=HOUSE_CATEGORIES,
        excluded_categories=None,
        mapper=None,
        is_vectorized=True,
    )
    ctx.add_stratification(
        name="familiar",
        sources=["familiar"],
        categories=FAMILIARS,
        excluded_categories=None,
        mapper=None,
        is_vectorized=True,
    )
    ctx.register_observation(
        observation_type=AddingObservation,
        name="this_shouldnt_work",
        pop_filter="",
        aggregator_sources=[],
        aggregator=sum,
        stratifications=("house", "height"),  # `height` is not a stratification
        when=lifecycle_phase,
        results_formatter=lambda: None,
    )

    with pytest.raises(KeyError, match="height"):
        for result, _measure, _updater in ctx.gather_results(
            population, lifecycle_phase, mocked_event
        ):
            print(result)


@pytest.mark.parametrize(
    "pop_filter, stratifications",
    [
        ('familiar=="cat"', tuple()),
        ('familiar=="spaghetti_yeti"', tuple()),
        ("", ("new_col1",)),
        ("", ("new_col1", "new_col2")),
        ('familiar=="cat"', ("new_col1",)),
        ("", tuple()),
    ],
    ids=[
        "pop_filter",
        "pop_filter_empties_dataframe",
        "single_excluded_stratification",
        "two_excluded_stratifications",
        "pop_filter_and_excluded_stratification",
        "no_pop_filter_or_excluded_stratifications",
    ],
)
def test__filter_population(pop_filter, stratifications):
    population = BASE_POPULATION.copy()
    if stratifications:
        # Make some of the stratifications missing to mimic mapping to excluded categories
        population["new_col1"] = "new_value1"
        population.loc[population["tracked"] == True, "new_col1"] = np.nan
        if len(stratifications) == 2:
            population["new_col2"] = "new_value2"
            population.loc[population["new_col1"].notna(), "new_col2"] = np.nan
        # Add on the post-stratified columns
        for stratification in stratifications:
            mapped_col = f"{stratification}_mapped_values"
            population[mapped_col] = population[stratification]

    filtered_pop = ResultsContext()._filter_population(
        population=population, pop_filter=pop_filter, stratification_names=stratifications
    )
    expected = population.copy()
    if pop_filter:
        familiar = pop_filter.split("==")[1].strip('"')
        expected = expected[expected["familiar"] == familiar]
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
def test__get_groups(stratifications, values):

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


def test_to_observe(mocked_event, mocker):
    """Test that to_observe can be used to turn off observations"""
    ctx = ResultsContext()

    # Generate population DataFrame
    population = BASE_POPULATION.copy()

    lifecycle_phase = "collect_metrics"
    ctx.register_observation(
        observation_type=AddingObservation,
        name="wizard_count",
        pop_filter="house == 'hufflepuff'",
        aggregator_sources=[],
        aggregator=len,
        stratifications=tuple(),
        when=lifecycle_phase,
        results_formatter=lambda: None,
    )

    for result, _measure, _updater in ctx.gather_results(
        population, lifecycle_phase, mocked_event
    ):
        assert not result.empty

    # Extract the observation from the context and patch it to not observe
    observation = list(ctx.observations["collect_metrics"].values())[0][0]
    mocker.patch.object(observation, "to_observe", return_value=False)
    for result, _measure, _updater in ctx.gather_results(
        population, lifecycle_phase, mocked_event
    ):
        assert not result
