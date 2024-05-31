import itertools
import math
from datetime import timedelta

import pandas as pd
import pytest
from pandas.core.groupby import DataFrameGroupBy

from tests.framework.results.helpers import (
    BASE_POPULATION,
    CATEGORIES,
    FAMILIARS,
    NAME,
    SOURCES,
    sorting_hat_serial,
    sorting_hat_vector,
    verify_stratification_added,
)
from vivarium.framework.results import VALUE_COLUMN
from vivarium.framework.results.context import ResultsContext


@pytest.mark.parametrize(
    "name, sources, categories, mapper, is_vectorized",
    [
        (NAME, SOURCES, CATEGORIES, sorting_hat_vector, True),
        (NAME, SOURCES, CATEGORIES, sorting_hat_serial, False),
    ],
    ids=["vectorized_mapper", "non-vectorized_mapper"],
)
def test_add_stratification(name, sources, categories, mapper, is_vectorized):
    ctx = ResultsContext()
    assert not verify_stratification_added(
        ctx.stratifications, name, sources, categories, mapper, is_vectorized
    )
    ctx.add_stratification(name, sources, categories, mapper, is_vectorized)
    assert verify_stratification_added(
        ctx.stratifications, name, sources, categories, mapper, is_vectorized
    )


@pytest.mark.parametrize(
    "name, sources, categories, mapper, is_vectorized, expected_exception",
    [
        (  # sources not in population columns
            NAME,
            ["middle_initial"],
            CATEGORIES,
            sorting_hat_vector,
            True,
            TypeError,
        ),
        (  # is_vectorized=True with non-vectorized mapper
            NAME,
            SOURCES,
            CATEGORIES,
            sorting_hat_serial,
            True,
            Exception,
        ),
        (  # is_vectorized=False with vectorized mapper
            NAME,
            SOURCES,
            CATEGORIES,
            sorting_hat_vector,
            False,
            Exception,
        ),
    ],
)
def test_add_stratification_raises(
    name, sources, categories, mapper, is_vectorized, expected_exception
):
    ctx = ResultsContext()
    with pytest.raises(expected_exception):
        raise ctx.add_stratification(name, sources, categories, mapper, is_vectorized)


def test_add_stratifcation_duplicate_name_raises():
    ctx = ResultsContext()
    ctx.add_stratification(NAME, SOURCES, CATEGORIES, sorting_hat_vector, True)
    with pytest.raises(ValueError, match=f"Stratification name '{NAME}' is already used: "):
        # register a different stratification but w/ the same name
        ctx.add_stratification(NAME, [], [], None, False)


def _aggregate_state_person_time(x: pd.DataFrame) -> float:
    """Helper aggregator function for observation testing"""
    return len(x) * (28 / 365.25)


@pytest.mark.parametrize(
    "name, pop_filter, aggregator, additional_stratifications, excluded_stratifications, when",
    [
        (
            "living_person_time",
            'alive == "alive" and undead == False',
            _aggregate_state_person_time,
            [],
            [],
            "collect_metrics",
        ),
        (
            "undead_person_time",
            "undead == True",
            _aggregate_state_person_time,
            [],
            [],
            "time_step__prepare",
        ),
    ],
    ids=["valid_on_collect_metrics", "valid_on_time_step__prepare"],
)
def test_add_observation(
    name, pop_filter, aggregator, additional_stratifications, excluded_stratifications, when
):
    ctx = ResultsContext()
    ctx._default_stratifications = ["age", "sex"]
    assert len(ctx.observations) == 0
    ctx.add_observation(
        name=name,
        pop_filter=pop_filter,
        aggregator_sources=[],
        aggregator=aggregator,
        additional_stratifications=additional_stratifications,
        excluded_stratifications=excluded_stratifications,
        when=when,
        formatter=lambda: None,
    )
    assert len(ctx.observations) == 1


@pytest.mark.parametrize(
    "name, pop_filter, aggregator, additional_stratifications, excluded_stratifications, when",
    [
        (
            "living_person_time",
            'alive == "alive" and undead == False',
            _aggregate_state_person_time,
            [],
            [],
            "collect_metrics",
        ),
    ],
    ids=["valid_on_collect_metrics"],
)
def test_double_add_observation(
    name, pop_filter, aggregator, additional_stratifications, excluded_stratifications, when
):
    """Tests a double add of the same stratification, this should result in one
    additional observation being added to the context."""
    ctx = ResultsContext()
    ctx._default_stratifications = ["age", "sex"]
    assert len(ctx.observations) == 0
    ctx.add_observation(
        name=name,
        pop_filter=pop_filter,
        aggregator_sources=[],
        aggregator=aggregator,
        additional_stratifications=additional_stratifications,
        excluded_stratifications=excluded_stratifications,
        when=when,
        formatter=lambda: None,
    )
    ctx.add_observation(
        name=name,
        pop_filter=pop_filter,
        aggregator_sources=[],
        aggregator=aggregator,
        additional_stratifications=additional_stratifications,
        excluded_stratifications=excluded_stratifications,
        when=when,
        formatter=lambda: None,
    )
    assert len(ctx.observations) == 1


@pytest.mark.parametrize(
    "default_stratifications, additional_stratifications, excluded_stratifications, expected_stratifications",
    [
        ([], [], [], ()),
        (["age", "sex"], ["handedness"], ["age"], ("sex", "handedness")),
        (["age", "sex"], [], ["age", "sex"], ()),
        (["age"], [], ["bogus_exclude_column"], ("age",)),
    ],
    ids=[
        "empty_add_empty_exclude",
        "one_add_one_exclude",
        "all_defaults_excluded",
        "bogus_exclude",
    ],
)
def test__get_stratifications(
    default_stratifications,
    additional_stratifications,
    excluded_stratifications,
    expected_stratifications,
):
    ctx = ResultsContext()
    # default_stratifications would normally be set via ResultsInterface.set_default_stratifications()
    ctx.default_stratifications = default_stratifications
    stratifications = ctx._get_stratifications(
        additional_stratifications, excluded_stratifications
    )
    assert sorted(stratifications) == sorted(expected_stratifications)


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
def test_gather_results(pop_filter, aggregator_sources, aggregator, stratifications):
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
    event_name = "collect_metrics"

    # Set up stratifications
    if "house" in stratifications:
        ctx.add_stratification("house", ["house"], CATEGORIES, None, True)
    if "familiar" in stratifications:
        ctx.add_stratification("familiar", ["familiar"], FAMILIARS, None, True)
    ctx.add_observation(
        name="foo",
        pop_filter=pop_filter,
        aggregator_sources=aggregator_sources,
        aggregator=aggregator,
        additional_stratifications=stratifications,
        excluded_stratifications=[],
        when=event_name,
        formatter=lambda: None,
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
    for result, _measure in ctx.gather_results(population, event_name):
        assert all(
            math.isclose(actual_result, expected_result, rel_tol=0.0001)
            for actual_result in result.values
        )
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
    name, pop_filter, aggregator_sources, aggregator, stratifications
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

    event_name = "collect_metrics"

    # Set up stratifications
    if "house" in stratifications:
        ctx.add_stratification("house", ["house"], CATEGORIES, None, True)
    if "familiar" in stratifications:
        ctx.add_stratification("familiar", ["familiar"], FAMILIARS, None, True)
    ctx.add_observation(
        name=name,
        pop_filter=pop_filter,
        aggregator_sources=aggregator_sources,
        aggregator=aggregator,
        additional_stratifications=stratifications,
        excluded_stratifications=[],
        when=event_name,
        formatter=lambda: None,
    )

    for results, _measure in ctx.gather_results(population, event_name):
        unladen_results = results.loc["unladen_swallow"]
        assert len(unladen_results) > 0
        assert (unladen_results[VALUE_COLUMN] == 0).all()


def test_gather_results_with_empty_pop_filter():
    """Test case where pop_filter filters to an empty population. gather_results
    should return None.
    """
    ctx = ResultsContext()

    # Generate population DataFrame
    population = BASE_POPULATION.copy()

    event_name = "collect_metrics"
    ctx.add_observation(
        name="wizard_count",
        pop_filter="house == 'durmstrang'",
        aggregator_sources=[],
        aggregator=len,
        additional_stratifications=[],
        excluded_stratifications=[],
        when=event_name,
        formatter=lambda: None,
    )

    for result, _measure in ctx.gather_results(population, event_name):
        assert not result


def test_gather_results_with_no_stratifications():
    """Test case where we have no stratifications. gather_results should return one value."""
    ctx = ResultsContext()

    # Generate population DataFrame
    population = BASE_POPULATION.copy()

    event_name = "collect_metrics"
    ctx.add_observation(
        name="wizard_count",
        pop_filter="",
        aggregator_sources=None,
        aggregator=len,
        additional_stratifications=[],
        excluded_stratifications=[],
        when=event_name,
        formatter=lambda: None,
    )

    assert len(ctx.stratifications) == 0
    assert (
        len(list(result for result, _measure in ctx.gather_results(population, event_name)))
        == 1
    )


def test_bad_aggregator_stratification():
    """Test if an exception gets raised when a stratification that doesn't
    exist is attempted to be used, as expected."""
    ctx = ResultsContext()

    # Generate population DataFrame
    population = BASE_POPULATION.copy()
    event_name = "collect_metrics"

    # Set up stratifications
    ctx.add_stratification("house", ["house"], CATEGORIES, None, True)
    ctx.add_stratification("familiar", ["familiar"], FAMILIARS, None, True)
    ctx.add_observation(
        name="this_shouldnt_work",
        pop_filter="",
        aggregator_sources=[],
        aggregator=sum,
        additional_stratifications=["house", "height"],  # `height` is not a stratification
        excluded_stratifications=[],
        when=event_name,
        formatter=lambda: None,
    )

    with pytest.raises(KeyError, match="height"):
        for result, _measure in ctx.gather_results(population, event_name):
            print(result)


@pytest.mark.parametrize(
    "pop_filter",
    [
        'familiar=="spaghetti_yeti"',
        'familiar=="cat"',
        "",
    ],
)
def test__filter_population(pop_filter):
    filtered_pop = ResultsContext()._filter_population(
        population=BASE_POPULATION, pop_filter=pop_filter
    )
    if pop_filter:
        familiar = pop_filter.split("==")[1].strip('"')
        assert filtered_pop.equals(BASE_POPULATION[BASE_POPULATION["familiar"] == familiar])
        if not familiar in filtered_pop["familiar"].values:
            assert filtered_pop.empty
    else:
        # An empty pop filter should return the entire population
        assert filtered_pop.equals(BASE_POPULATION)


@pytest.mark.parametrize(
    "stratifications, values",
    [
        (("familiar",), [FAMILIARS]),
        (("familiar", "house"), [FAMILIARS, CATEGORIES]),
        ((), ["all"]),
    ],
)
def test__get_groups(stratifications, values):
    groups = ResultsContext()._get_groups(
        stratifications=stratifications, filtered_pop=BASE_POPULATION
    )
    combinations = set(itertools.product(*values))
    if len(values) == 1:
        # convert from set of tuples to set of strings
        combinations = set([comb[0] for comb in combinations])
    assert isinstance(groups, DataFrameGroupBy)
    if stratifications:
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
def test__aggregate(stratifications, aggregator_sources, aggregator):
    """Test that we are aggregating correctly. There are some nuances here:
    - If aggregator_resources is provided, then simply .apply it to the groups passed in.
    - If no aggregator_resources are provided, then we want a full aggregation of the groups.
    - _aggregate can return either a pd.Series or a pd.DataFrame of any number of columns
    """
    groups = ResultsContext()._get_groups(
        stratifications=stratifications, filtered_pop=BASE_POPULATION
    )
    aggregates = ResultsContext()._aggregate(
        pop_groups=groups,
        aggregator_sources=aggregator_sources,
        aggregator=aggregator,
    )
    if aggregator == len:
        if stratifications:
            stratification_idx = (
                set(itertools.product(*(FAMILIARS, CATEGORIES)))
                if "house" in stratifications
                else set(FAMILIARS)
            )
            assert set(aggregates.index) == stratification_idx
            assert (aggregates.values == len(BASE_POPULATION) / groups.ngroups).all()
        else:
            assert len(aggregates.values) == 1
            assert aggregates.values[0] == len(BASE_POPULATION)
    else:  # sum aggregator
        assert aggregates.shape[1] == 2
        expected = BASE_POPULATION[["power_level", "tracked"]].sum() / groups.ngroups
        if stratifications:
            stratification_idx = (
                set(itertools.product(*(FAMILIARS, CATEGORIES)))
                if "house" in stratifications
                else set(FAMILIARS)
            )
            assert set(aggregates.index) == stratification_idx
            assert (aggregates.sum() / groups.ngroups).equals(expected)
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
def test__format(aggregates):
    new_aggregates = ResultsContext()._format(aggregates=aggregates)
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
def test__expand_index(aggregates):
    full_idx_aggregates = ResultsContext()._expand_index(aggregates=aggregates)
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
