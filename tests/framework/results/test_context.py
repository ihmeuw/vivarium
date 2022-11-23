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
        ctx._stratifications, name, sources, categories, mapper, is_vectorized
    )
    ctx.add_stratification(name, sources, categories, mapper, is_vectorized)
    assert verify_stratification_added(
        ctx._stratifications, name, sources, categories, mapper, is_vectorized
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


def _aggregate_state_person_time(self, x: pd.DataFrame) -> float:
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
    assert len(ctx._observations) == 0
    ctx.add_observation(
        name,
        pop_filter,
        aggregator,
        additional_stratifications,
        excluded_stratifications,
        when,
    )
    assert len(ctx._observations) == 1


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
    assert len(ctx._observations) == 0
    ctx.add_observation(
        name,
        pop_filter,
        aggregator,
        additional_stratifications,
        excluded_stratifications,
        when,
    )
    ctx.add_observation(
        name,
        pop_filter,
        aggregator,
        additional_stratifications,
        excluded_stratifications,
        when,
    )
    assert len(ctx._observations) == 1


@pytest.mark.parametrize(
    "default, additional, excluded, match",
    [
        (["age", "sex"], ["age"], [], "age"),
        (["age", "sex"], [], ["eye_color"], "eye_color"),
        (["age", "sex"], ["age"], ["eye_color"], "age|eye_color"),
    ],
    ids=[
        "additional_no_operation",
        "exclude_no_operation",
        "additional_and_exclude_no_operation",
    ],
)
def test_add_observation_nop_stratifications(default, additional, excluded, match):
    ctx = ResultsContext()
    ctx._default_stratifications = default
    with pytest.warns(UserWarning, match=match):
        ctx.add_observation(
            "name",
            'alive == "alive"',
            _aggregate_state_person_time,
            additional,
            excluded,
            "collect_metrics",
        )


@pytest.mark.parametrize(
    "default_stratifications, additional_stratifications, excluded_stratifications, expected_groupers",
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
def test__get_groupers(
    default_stratifications,
    additional_stratifications,
    excluded_stratifications,
    expected_groupers,
):
    ctx = ResultsContext()
    # default_stratifications would normally be set via ResultsInterface.set_default_stratifications()
    ctx._default_stratifications = default_stratifications
    groupers = ctx._get_groupers(additional_stratifications, excluded_stratifications)
    assert sorted(groupers) == sorted(expected_groupers)


def test_gather_results():
    ctx = ResultsContext()

    # Generate population DataFrame
    population = BASE_POPULATION
    population.drop(["tracked"], axis=1)
    # Mock out some extra columns that would be produced by the manager's _prepare_population() method
    population["current_time"] = pd.Timestamp(year=2045, month=1, day=1, hour=12)
    population["step_size"] = timedelta(days=28)
    population["event_time"] = pd.Timestamp(year=2045, month=1, day=1, hour=12) + timedelta(days=28)
    event_name = "collect_metrics"

    # Set up stratifications
    # XXX mek: should stratification names be unique against the sources? Are they simply cosmetic??
    ctx.add_stratification("house", ["house"], CATEGORIES, None, True)
    ctx.add_stratification("familiar", ["familiar"], FAMILIARS, None, True)
    ctx.add_observation("power_level", "tracked==True", len, ["house", "familiar"], [], "collect_metrics")

    i = 0
    for r in ctx.gather_results(population, "collect_metrics"):
        print(r)
        i += 1
    assert i == 1


def test__format_results():
    # TODO: do real tests
    ctx = ResultsContext()
    rv = ctx._format_results()

    assert True
