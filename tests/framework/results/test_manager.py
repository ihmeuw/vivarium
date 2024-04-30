import re
from types import MethodType

import pandas as pd
import pytest
from loguru import logger

from tests.framework.results.helpers import (
    BIN_BINNED_COLUMN,
    BIN_LABELS,
    BIN_SILLY_BINS,
    BIN_SOURCE,
    CATEGORIES,
    CONFIG,
    FAMILIARS,
    NAME,
    POWER_LEVELS,
    SOURCES,
    STUDENT_HOUSES,
    FullyFilteredHousePointsObserver,
    Hogwarts,
    HogwartsResultsStratifier,
    HousePointsObserver,
    NoStratificationsQuidditchWinsObserver,
    QuidditchWinsObserver,
    mock_get_value,
    sorting_hat_serial,
    sorting_hat_vector,
    verify_stratification_added,
)
from vivarium.framework.results.manager import ResultsManager
from vivarium.interface.interactive import InteractiveContext

#######################################
# Tests for `register_stratification` #
#######################################


@pytest.mark.parametrize(
    "name, sources, categories, mapper, is_vectorized",
    [
        (
            NAME,
            SOURCES,
            CATEGORIES,
            sorting_hat_vector,
            True,
        ),
        (
            NAME,
            SOURCES,
            CATEGORIES,
            sorting_hat_serial,
            False,
        ),
    ],
    ids=["vectorized_mapper", "non-vectorized_mapper"],
)
def test_register_stratification_no_pipelines(
    name, sources, categories, mapper, is_vectorized, mocker
):
    mgr = ResultsManager()
    builder = mocker.Mock()
    mgr.setup(builder)
    mgr.register_stratification(name, categories, mapper, is_vectorized, sources, [])
    for item in sources:
        assert item in mgr._required_columns
    assert verify_stratification_added(
        mgr._results_context.stratifications,
        name,
        sources,
        categories,
        mapper,
        is_vectorized,
    )


@pytest.mark.parametrize(
    "name, sources, categories, mapper, is_vectorized",
    [
        (
            NAME,
            SOURCES,
            CATEGORIES,
            sorting_hat_vector,
            True,
        ),
        (
            NAME,
            SOURCES,
            CATEGORIES,
            sorting_hat_serial,
            False,
        ),
    ],
    ids=["vectorized_mapper", "non-vectorized_mapper"],
)
def test_register_stratification_with_pipelines(
    name, sources, categories, mapper, is_vectorized, mocker
):
    mgr = ResultsManager()
    builder = mocker.Mock()
    # Set up mock builder with mocked get_value call for Pipelines
    mocker.patch.object(builder, "value.get_value")
    builder.value.get_value = MethodType(mock_get_value, builder)
    mgr.setup(builder)
    mgr.register_stratification(name, categories, mapper, is_vectorized, [], sources)
    for item in sources:
        assert item in mgr._required_values
    assert verify_stratification_added(
        mgr._results_context.stratifications,
        name,
        sources,
        categories,
        mapper,
        is_vectorized,
    )


@pytest.mark.parametrize(
    "name, sources, categories, mapper, is_vectorized",
    [
        (  # expected Stratification for vectorized
            NAME,
            SOURCES,
            CATEGORIES,
            sorting_hat_vector,
            True,
        ),
        (  # expected Stratification for non-vectorized
            NAME,
            SOURCES,
            CATEGORIES,
            sorting_hat_serial,
            False,
        ),
    ],
    ids=["vectorized_mapper", "non-vectorized_mapper"],
)
def test_register_stratification_with_column_and_pipelines(
    name, sources, categories, mapper, is_vectorized, mocker
):
    mgr = ResultsManager()
    builder = mocker.Mock()
    # Set up mock builder with mocked get_value call for Pipelines
    mocker.patch.object(builder, "value.get_value")
    builder.value.get_value = MethodType(mock_get_value, builder)
    mgr.setup(builder)
    mocked_column_name = "silly_column"
    mgr.register_stratification(
        name, categories, mapper, is_vectorized, [mocked_column_name], sources
    )
    assert mocked_column_name in mgr._required_columns
    for item in sources:
        assert item in mgr._required_values
    all_sources = sources.copy()
    all_sources.append(mocked_column_name)
    assert verify_stratification_added(
        mgr._results_context.stratifications,
        name,
        all_sources,
        categories,
        mapper,
        is_vectorized,
    )


def test_duplicate_name_register_stratification(mocker):
    mgr = ResultsManager()
    builder = mocker.Mock()
    mgr.setup(builder)
    mgr.register_stratification(NAME, CATEGORIES, sorting_hat_serial, False, SOURCES, [])
    with pytest.raises(ValueError, match=f"Name `{NAME}` is already used"):
        mgr.register_stratification(NAME, CATEGORIES, sorting_hat_vector, True, SOURCES, [])


##############################################
# Tests for `register_binned_stratification` #
##############################################


def test_register_binned_stratification(mocker):
    mgr = ResultsManager()
    mock_register_stratification = mocker.patch(
        "vivarium.framework.results.manager.ResultsManager.register_stratification"
    )
    mgr.register_binned_stratification(
        BIN_SOURCE, "column", BIN_BINNED_COLUMN, BIN_SILLY_BINS, BIN_LABELS
    )
    mock_register_stratification.assert_called_once()


@pytest.mark.parametrize(
    "bins, labels",
    [(BIN_SILLY_BINS, BIN_LABELS[2:]), (BIN_SILLY_BINS[2:], BIN_LABELS)],
    ids=["more_bins_than_labels", "more_labels_than_bins"],
)
def test_register_binned_stratification_raises(bins, labels):
    mgr = ResultsManager()
    with pytest.raises(ValueError):
        raise mgr.register_binned_stratification(
            BIN_SOURCE, "column", BIN_BINNED_COLUMN, bins, labels
        )


@pytest.mark.parametrize(
    "default, additional, excluded, match",
    [
        (["age", "sex"], ["age"], [], ["age"]),
        (["age", "sex"], [], ["eye_color"], ["eye_color"]),
        (["age", "sex"], ["age"], ["eye_color"], ["age", "eye_color"]),
    ],
    ids=[
        "additional_no_operation",
        "exclude_no_operation",
        "additional_and_exclude_no_operation",
    ],
)
def test_add_observation_nop_stratifications(
    default, additional, excluded, match, mocker, caplog
):
    mgr = ResultsManager()
    builder = mocker.Mock()
    mgr.setup(builder)
    mgr.logger = logger

    mgr._results_context.default_stratifications = default
    mgr.register_observation(
        name="name",
        pop_filter='alive == "alive"',
        aggregator_sources=[],
        aggregator=lambda: None,
        requires_columns=[],
        requires_values=[],
        additional_stratifications=additional,
        excluded_stratifications=excluded,
        when="collect_metrics",
        report=lambda: None,
    )
    for m in match:
        assert m in caplog.text


def test_setting_default_stratifications_at_setup(mocker):
    """Test that set default stratifications happens at setup"""
    mgr = ResultsManager()
    builder = mocker.Mock()
    mgr._results_context.set_default_stratifications = mocker.Mock()
    mgr._results_context.set_default_stratifications.assert_not_called()

    mgr.setup(builder)

    mgr._results_context.set_default_stratifications.assert_called_once_with(
        builder.configuration.stratification.default
    )


################################
# Tests for results processing #
################################


def test_metrics_initialized_as_empty_dict(mocker):
    """Test that metrics are initialized as an empty dictionary"""
    mgr = ResultsManager()
    builder = mocker.Mock()
    mgr.setup(builder)
    assert mgr.metrics == {}


def test_stratified_metrics_initialization():
    """Test that matrics are being initialized correctly. We expect a dictionary
    of pd.DataFrames. Each key of the dictionary is an observed measure name and
    the corresponding value is a zeroed-out multiindex pd.DataFrame of that observer's
    stratifications.
    """

    components = [
        HousePointsObserver(),
        QuidditchWinsObserver(),
        HogwartsResultsStratifier(),
    ]

    sim = InteractiveContext(configuration=CONFIG, components=components)
    metrics = sim._results.metrics
    assert isinstance(metrics, dict)
    assert set(metrics) == set(["house_points", "quidditch_wins"])
    for metric in metrics:
        result = metrics[metric]
        assert isinstance(result, pd.DataFrame)
        assert (result["value"] == 0).all()
    STUDENT_HOUSES_LIST = list(STUDENT_HOUSES)
    POWER_LEVELS_STR = [str(lvl) for lvl in POWER_LEVELS]
    assert metrics["house_points"].index.equals(
        pd.MultiIndex.from_product(
            [STUDENT_HOUSES_LIST, POWER_LEVELS_STR],
            names=["student_house", "power_level"],
        )
    )
    assert metrics["quidditch_wins"].index.equals(
        pd.MultiIndex.from_product(
            [FAMILIARS, POWER_LEVELS_STR],
            names=["familiar", "power_level"],
        )
    )


def test_metrics_initialized_from_no_stratifications_observer():
    """Test that Observers requesting no stratifications result in a
    single-row DataFrame with 'value' of zero and index labeled 'all'
    """
    components = [Hogwarts(), NoStratificationsQuidditchWinsObserver()]
    sim = InteractiveContext(configuration=CONFIG, components=components)
    results = sim._results.metrics["no_stratifications_quidditch_wins"]
    assert isinstance(results, pd.DataFrame)
    assert results.shape == (1, 1)
    assert results["value"].iat[0] == 0
    assert results.index.equals(pd.Index(["all"]))


def test_observers_with_missing_stratifications_fail():
    """Test that an error is raised if an Observer requests a stratification
    that never actually gets registered.
    """
    components = [QuidditchWinsObserver(), HousePointsObserver(), Hogwarts()]

    expected_missing = {  # NOTE: keep in alphabetical order
        "house_points": ["power_level", "student_house"],
        "quidditch_wins": ["familiar", "power_level"],
    }
    expected_log_msg = re.escape(
        "The following Observers are requested to be stratified by Stratifications "
        f"that are not registered: {expected_missing}"
    )

    with pytest.raises(ValueError, match=expected_log_msg):
        InteractiveContext(configuration=CONFIG, components=components)


def test_unused_stratifications_are_logged(caplog):
    """Test that we issue a logger.info warning if Stratifications are registered
    but never actually used by an Observer

    The HogwartsResultsStratifier registers "student_house", "familiar", and
    "power_level" stratifiers. However, we will only use the HousePointsObserver
    component which only requests to be stratified by "student_house" and "power_level"
    """
    components = [HousePointsObserver(), Hogwarts(), HogwartsResultsStratifier()]
    InteractiveContext(configuration=CONFIG, components=components)

    log_split = caplog.text.split(
        "The following Stratifications are registered but not used by any Observers: \n"
    )
    # Check that the log message is present and only exists one time
    assert len(log_split) == 2
    # Check that the log message contains the expected Stratifications
    assert "['familiar']" in log_split[1]


def test_update_monotonically_increasing_metrics():
    """Test that (monotonically increasing) metrics are being updated correctly."""

    def _check_house_points(pop: pd.DataFrame, step_number: int) -> None:
        """We know that house points are stratified by 'student_house' and 'power_level'.
        and that each wizard of gryffindor and of level 50 and 80 gains a point
        """
        assert set(pop["house_points"]) == set([0, 1])
        assert (pop.loc[pop["house_points"] != 0, "student_house"] == "gryffindor").all()
        assert set(pop.loc[pop["house_points"] != 0, "power_level"]) == set(["50", "80"])
        group_sizes = pd.DataFrame(
            pop.groupby(["student_house", "power_level"]).size().astype("float"),
            columns=["value"],
        )
        metrics = sim._results.metrics["house_points"]
        assert metrics[metrics["value"] != 0].equals(
            group_sizes.loc(axis=0)["gryffindor", ["50", "80"]] * step_number
        )

    def _check_quidditch_wins(pop: pd.DataFrame, step_number: int) -> None:
        """We know that quidditch wins are stratified by 'familiar' and 'power_level'.
        and that each wizard with a banana slug familiar gains a point
        """
        assert set(pop["quidditch_wins"]) == set([0, 1])
        assert (pop.loc[pop["quidditch_wins"] != 0, "familiar"] == "banana_slug").all()
        group_sizes = pd.DataFrame(
            pop.groupby(["familiar", "power_level"]).size().astype("float"), columns=["value"]
        )
        metrics = sim._results.metrics["quidditch_wins"]
        assert metrics[metrics["value"] != 0].equals(
            group_sizes[group_sizes.index.get_level_values(0) == "banana_slug"] * step_number
        )

    components = [
        Hogwarts(),
        HousePointsObserver(),
        QuidditchWinsObserver(),
        HogwartsResultsStratifier(),
    ]
    sim = InteractiveContext(configuration=CONFIG, components=components)
    sim.step()
    pop = sim.get_population()
    _check_house_points(pop, step_number=1)
    _check_quidditch_wins(pop, step_number=1)

    sim.step()
    pop = sim.get_population()
    _check_house_points(pop, step_number=2)
    _check_quidditch_wins(pop, step_number=2)


def test_update_metrics_fully_filtered_pop():
    components = [
        Hogwarts(),
        FullyFilteredHousePointsObserver(),
        HogwartsResultsStratifier(),
    ]
    sim = InteractiveContext(configuration=CONFIG, components=components)
    sim.step()
    # The FullyFilteredHousePointsObserver filters the population to a bogus
    # power level and so we should not be observing anything
    assert (sim._results.metrics["house_points"]["value"] == 0).all()
    sim.step()
    assert (sim._results.metrics["house_points"]["value"] == 0).all()


def test_update_metrics_no_stratifications():
    components = [Hogwarts(), NoStratificationsQuidditchWinsObserver()]
    sim = InteractiveContext(configuration=CONFIG, components=components)
    sim.step()
    pop = sim.get_population()
    results = sim._results.metrics["no_stratifications_quidditch_wins"]
    assert results.loc["all"]["value"] == pop["quidditch_wins"].sum()
    sim.step()
    pop = sim.get_population()
    results = sim._results.metrics["no_stratifications_quidditch_wins"]
    assert results.loc["all"]["value"] == pop["quidditch_wins"].sum() * 2
