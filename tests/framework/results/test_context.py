import math
from datetime import timedelta

import pandas as pd
import pytest

from vivarium.framework.results.context import ResultsContext

from .mocks import (
    BASE_POPULATION,
    CATEGORIES,
    FAMILIARS,
    NAME,
    SOURCES,
    sorting_hat_serial,
    sorting_hat_vector,
    verify_stratification_added,
)


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


def _aggregate_state_person_time(x: pd.DataFrame) -> float:
    """Helper aggregator function for observation testing"""
    return len(x) * (28 / 365.35)


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
        name,
        pop_filter,
        [],
        aggregator,
        additional_stratifications,
        excluded_stratifications,
        when,
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
    """Tests a double add of the same stratification, this should result in one additional observation being added to
    the context."""
    ctx = ResultsContext()
    ctx._default_stratifications = ["age", "sex"]
    assert len(ctx.observations) == 0
    ctx.add_observation(
        name,
        pop_filter,
        [],
        aggregator,
        additional_stratifications,
        excluded_stratifications,
        when,
    )
    ctx.add_observation(
        name,
        pop_filter,
        [],
        aggregator,
        additional_stratifications,
        excluded_stratifications,
        when,
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
    "name, pop_filter, aggregator_sources, aggregator, stratifications, expected_result",
    [
        ("wizard_count", "tracked==True", None, len, ["house", "familiar"], 4),
        (
            "power_level_total",
            "tracked==True",
            ["power_level"],
            sum,
            ["house", "familiar"],
            260,
        ),
        (
            "wizard_time",
            "tracked==True",
            [],
            _aggregate_state_person_time,
            ["house", "familiar"],
            0.306555,
        ),
        ("wizard_count", "tracked==True", None, len, ["house"], 20),
        ("power_level_total", "tracked==True", ["power_level"], sum, ["house"], 1300),
        (
            "wizard_time",
            "tracked==True",
            [],
            _aggregate_state_person_time,
            ["house"],
            1.53277,
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
def test_gather_results(
    name, pop_filter, aggregator_sources, aggregator, stratifications, expected_result
):
    """Test cases where every stratification is in gather_results. Checks for existence and correctness
    of results"""
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
        name,
        pop_filter,
        aggregator_sources,
        aggregator,
        stratifications,
        [],
        event_name,
    )

    i = 0
    for r in ctx.gather_results(population, event_name):
        assert all(
            math.isclose(result, expected_result, rel_tol=0.0001) for result in r.values()
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
        name,
        pop_filter,
        aggregator_sources,
        aggregator,
        stratifications,
        [],
        event_name,
    )

    for r in ctx.gather_results(population, event_name):
        unladen_results = {k: v for (k, v) in r.items() if "unladen_swallow" in k}
        assert len(unladen_results.items()) > 0
        assert all(v == 0 for v in unladen_results.values())


def test_gather_results_with_empty_pop_filter():
    """Test case where pop_filter filters to an empty population. gather_results should return an empty dict"""
    ctx = ResultsContext()

    # Generate population DataFrame
    population = BASE_POPULATION.copy()

    event_name = "collect_metrics"
    ctx.add_observation(
        name="wizard_count",
        pop_filter="house == 'durmstrang'",
        aggregator_sources=None,
        aggregator=len,
        event_name=event_name,
    )

    for result in ctx.gather_results(population, event_name):
        assert len(result) == 0


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
        event_name=event_name,
    )

    assert len(ctx.stratifications) == 0
    assert len(list(ctx.gather_results(population, event_name))) == 1


def test__format_results():
    """Test that format results produces the expected number of keys and a specific expected key"""
    ctx = ResultsContext()
    aggregates = BASE_POPULATION.groupby(["house", "familiar"]).apply(len)
    measure = "wizard_count"
    rv = ctx._format_results(measure, aggregates, has_stratifications=True)

    # Check that the number of expected data column names are there
    expected_keys_len = len(CATEGORIES) * len(FAMILIARS)
    assert len(rv.keys()) == expected_keys_len

    # Check that an example data column name is there
    expected_key = "MEASURE_wizard_count_HOUSE_slytherin_FAMILIAR_cat"
    assert expected_key in rv.keys()


def test__bad_aggregator_return():
    """Test that an exception is raised, as expected, when an aggregator
    produces something other than a pd.DataFrame with a single column or a pd.Series"""
    ctx = ResultsContext()

    # Generate population DataFrame
    population = BASE_POPULATION.copy()
    event_name = "collect_metrics"

    # Set up stratifications
    ctx.add_stratification("house", ["house"], CATEGORIES, None, True)
    ctx.add_stratification("familiar", ["familiar"], FAMILIARS, None, True)
    ctx.add_observation(
        "this_shouldnt_work",
        "",
        ["tracked", "power_level"],
        sum,
        ["house", "familiar"],
        [],
        event_name,
    )

    with pytest.raises(TypeError):
        for r in ctx.gather_results(population, event_name):
            print(r)


def test__bad_aggregator_stratification():
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
        "this_shouldnt_work",
        "",
        [],
        sum,
        ["house", "height"],  # `height` is not a stratification
        [],
        event_name,
    )

    with pytest.raises(KeyError, match="height"):
        for r in ctx.gather_results(population, event_name):
            print(r)
