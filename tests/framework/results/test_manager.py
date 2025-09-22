import re
from types import MethodType

import numpy as np
import pandas as pd
import pytest
import pytest_mock
from _pytest.logging import LogCaptureFixture
from layered_config_tree.main import LayeredConfigTree
from loguru import logger
from pandas.api.types import CategoricalDtype

from tests.framework.results.helpers import (
    BIN_BINNED_COLUMN,
    BIN_LABELS,
    BIN_SILLY_BIN_EDGES,
    BIN_SOURCE,
    FAMILIARS,
    HARRY_POTTER_CONFIG,
    HOUSE_CATEGORIES,
    NAME,
    NAME_COLUMNS,
    POWER_LEVEL_GROUP_LABELS,
    STUDENT_HOUSES,
    CatBombObserver,
    ExamScoreObserver,
    FullyFilteredHousePointsObserver,
    Hogwarts,
    HogwartsResultsStratifier,
    HousePointsObserver,
    MagicalAttributesObserver,
    NeverObserver,
    NoStratificationsQuidditchWinsObserver,
    QuidditchWinsObserver,
    ValedictorianObserver,
    mock_get_value,
    sorting_hat_serial,
    sorting_hat_vectorized,
    verify_stratification_added,
)
from vivarium.framework.event import Event
from vivarium.framework.lifecycle import lifecycle_states
from vivarium.framework.results import VALUE_COLUMN
from vivarium.framework.results.context import ResultsContext
from vivarium.framework.results.manager import ResultsManager
from vivarium.framework.results.observation import AddingObservation, Observation
from vivarium.framework.results.stratification import Stratification, get_mapped_col_name
from vivarium.framework.values import Pipeline
from vivarium.interface.interactive import InteractiveContext
from vivarium.types import ScalarMapper, VectorMapper


@pytest.mark.parametrize(
    "stratifications, default_stratifications, additional_stratifications, excluded_stratifications, expected_stratifications",
    [
        ([], [], [], [], []),
        (
            [],
            ["age", "sex"],
            ["handedness"],
            ["age"],
            ["sex", "handedness"],
        ),
        ([], ["age", "sex"], [], ["age", "sex"], []),
        ([], ["age"], [], ["bogus_exclude_column"], ["age"]),
        (["custom"], ["age", "sex"], [], [], ["custom", "age", "sex"]),
    ],
    ids=[
        "empty_add_empty_exclude",
        "one_add_one_exclude",
        "all_defaults_excluded",
        "bogus_exclude",
        "custom_stratification",
    ],
)
def test__get_stratifications(
    stratifications: list[str],
    default_stratifications: list[str],
    additional_stratifications: list[str],
    excluded_stratifications: list[str],
    expected_stratifications: list[str],
    mocker: pytest_mock.MockFixture,
) -> None:
    ctx = ResultsContext()
    ctx.default_stratifications = default_stratifications
    mgr = ResultsManager()
    mgr.logger = mocker.Mock()
    mocker.patch.object(mgr, "_results_context", ctx)
    # default_stratifications would normally be set via ResultsInterface.set_default_stratifications()
    final_stratifications = mgr._get_stratifications(
        stratifications, additional_stratifications, excluded_stratifications
    )
    assert sorted(final_stratifications) == sorted(expected_stratifications)


#######################################
# Tests for `register_stratification` #
#######################################


@pytest.mark.parametrize(
    "excluded_categories, mapper, is_vectorized",
    [
        ([], sorting_hat_vectorized, True),
        ([], sorting_hat_serial, False),
        (["gryffindor"], sorting_hat_vectorized, True),
    ],
    ids=["vectorized_mapper", "non-vectorized_mapper", "excluded_categories"],
)
def test_register_stratification_no_pipelines(
    mocker: pytest_mock.MockFixture,
    excluded_categories: list[str],
    mapper: VectorMapper | ScalarMapper,
    is_vectorized: bool,
) -> None:
    mgr = ResultsManager()
    builder = mocker.Mock()
    builder.configuration.stratification = LayeredConfigTree(
        {"default": [], "excluded_categories": {}}
    )
    mgr.setup(builder)
    mgr.register_stratification(
        name=NAME,
        categories=HOUSE_CATEGORIES,
        excluded_categories=excluded_categories,
        mapper=mapper,
        is_vectorized=is_vectorized,
        requires_columns=NAME_COLUMNS,
        requires_values=[],
    )
    assert verify_stratification_added(
        stratifications=mgr._results_context.stratifications,
        name=NAME,
        requires_columns=NAME_COLUMNS,
        requires_values=[],
        categories=HOUSE_CATEGORIES,
        excluded_categories=excluded_categories,
        mapper=mapper,
        is_vectorized=is_vectorized,
    )


@pytest.mark.parametrize(
    "mapper, is_vectorized",
    [
        (sorting_hat_vectorized, True),
        (sorting_hat_serial, False),
    ],
    ids=["vectorized_mapper", "non-vectorized_mapper"],
)
def test_register_stratification_with_pipelines(
    mocker: pytest_mock.MockFixture, mapper: VectorMapper | ScalarMapper, is_vectorized: bool
) -> None:
    mgr = ResultsManager()
    builder = mocker.Mock()
    builder.configuration.stratification = LayeredConfigTree(
        {"default": [], "excluded_categories": {}}
    )
    # Set up mock builder with mocked get_value call for Pipelines
    mocker.patch.object(builder, "value.get_value")
    builder.value.get_value = MethodType(mock_get_value, builder)
    mgr.setup(builder)
    mgr.register_stratification(
        name=NAME,
        categories=HOUSE_CATEGORIES,
        excluded_categories=None,
        mapper=mapper,
        is_vectorized=is_vectorized,
        requires_columns=[],
        requires_values=NAME_COLUMNS,
    )

    assert verify_stratification_added(
        stratifications=mgr._results_context.stratifications,
        name=NAME,
        requires_columns=[],
        requires_values=[Pipeline(name) for name in NAME_COLUMNS],
        categories=HOUSE_CATEGORIES,
        excluded_categories=[],
        mapper=mapper,
        is_vectorized=is_vectorized,
    )


@pytest.mark.parametrize(
    "mapper, is_vectorized",
    [
        (sorting_hat_vectorized, True),
        (sorting_hat_serial, False),
    ],
    ids=["vectorized_mapper", "non-vectorized_mapper"],
)
def test_register_stratification_with_column_and_pipelines(
    mocker: pytest_mock.MockFixture, mapper: VectorMapper | ScalarMapper, is_vectorized: bool
) -> None:
    mgr = ResultsManager()
    builder = mocker.Mock()
    builder.configuration.stratification = LayeredConfigTree(
        {"default": [], "excluded_categories": {}}
    )
    # Set up mock builder with mocked get_value call for Pipelines
    mocker.patch.object(builder, "value.get_value")
    builder.value.get_value = MethodType(mock_get_value, builder)
    mgr.setup(builder)
    mocked_column_name = "silly_column"
    mgr.register_stratification(
        name=NAME,
        categories=HOUSE_CATEGORIES,
        excluded_categories=None,
        mapper=mapper,
        is_vectorized=is_vectorized,
        requires_columns=[mocked_column_name],
        requires_values=NAME_COLUMNS,
    )

    assert verify_stratification_added(
        stratifications=mgr._results_context.stratifications,
        name=NAME,
        requires_columns=[mocked_column_name],
        requires_values=[Pipeline(name) for name in NAME_COLUMNS],
        categories=HOUSE_CATEGORIES,
        excluded_categories=[],
        mapper=mapper,
        is_vectorized=is_vectorized,
    )


##############################################
# Tests for `register_binned_stratification` #
##############################################


@pytest.mark.parametrize(
    "bins, labels",
    [(BIN_SILLY_BIN_EDGES, BIN_LABELS[1:]), (BIN_SILLY_BIN_EDGES[1:], BIN_LABELS)],
    ids=["too_many_bins", "too_many_labels"],
)
def test_register_binned_stratification_raises_bins_labels_mismatch(
    bins: list[float], labels: list[str]
) -> None:
    mgr = ResultsManager()
    with pytest.raises(
        ValueError,
        match=r"The number of bin edges plus 1 \(\d+\) does not match the number of labels \(\d+\)",
    ):
        mgr.register_binned_stratification(
            target=BIN_SOURCE,
            binned_column=BIN_BINNED_COLUMN,
            bin_edges=bins,
            labels=labels,
            excluded_categories=None,
            target_type="column",
        )


from vivarium.types import VectorMapper


def test_binned_stratification_mapper() -> None:
    mgr = ResultsManager()
    mgr.logger = logger
    mgr.register_binned_stratification(
        target=BIN_SOURCE,
        binned_column=BIN_BINNED_COLUMN,
        bin_edges=BIN_SILLY_BIN_EDGES,
        labels=BIN_LABELS,
        excluded_categories=None,
        target_type="column",
    )
    strat = mgr._results_context.stratifications[BIN_BINNED_COLUMN]
    data = pd.DataFrame([-np.inf] + BIN_SILLY_BIN_EDGES + [np.inf])
    groups = strat.vectorized_mapper(data)
    expected_groups = pd.Series([np.nan] + BIN_LABELS + [np.nan] * 2)
    assert (groups.isna() == expected_groups.isna()).all()
    assert (groups[groups.notna()] == expected_groups[expected_groups.notna()]).all()


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
    default: list[str],
    additional: list[str],
    excluded: list[str],
    match: list[str],
    mocker: pytest_mock.MockFixture,
    caplog: LogCaptureFixture,
) -> None:
    mgr = ResultsManager()
    builder = mocker.Mock()
    mgr.setup(builder)
    mgr.logger = logger

    mgr._results_context.default_stratifications = default
    mgr.register_observation(
        observation_type=AddingObservation,
        name="name",
        pop_filter='alive == "alive"',
        aggregator_sources=[],
        aggregator=lambda: None,
        requires_columns=[],
        requires_values=[],
        additional_stratifications=additional,
        excluded_stratifications=excluded,
        when=lifecycle_states.COLLECT_METRICS,
        results_formatter=lambda: None,
    )
    for m in match:
        assert m in caplog.text


def test_setting_default_stratifications_at_setup(mocker: pytest_mock.MockFixture) -> None:
    """Test that set default stratifications happens at setup"""
    mgr = ResultsManager()
    builder = mocker.Mock()
    mocker.patch.object(mgr._results_context, "set_default_stratifications")
    mgr._results_context.set_default_stratifications.assert_not_called()  # type: ignore[attr-defined]

    mgr.setup(builder)

    mgr._results_context.set_default_stratifications.assert_called_once_with(  # type: ignore[attr-defined]
        builder.configuration.stratification.default
    )


################################
# Tests for results processing #
################################


def test__raw_results_initialized_as_empty_dict(mocker: pytest_mock.MockFixture) -> None:
    """Test that raw results are initialized as an empty dictionary"""
    mgr = ResultsManager()
    builder = mocker.Mock()
    mgr.setup(builder)
    assert mgr._raw_results == {}


def test_stratified__raw_results_initialization() -> None:
    """Test that raw results are being initialized correctly. We expect a dictionary
    of pd.DataFrames. Each key of the dictionary is an observed measure name and
    the corresponding value is a zeroed-out multiindex pd.DataFrame of that observer's
    stratifications.
    """

    components = [
        HousePointsObserver(),
        QuidditchWinsObserver(),
        HogwartsResultsStratifier(),
    ]

    sim = InteractiveContext(configuration=HARRY_POTTER_CONFIG, components=components)
    raw_results = sim._results._raw_results
    assert isinstance(raw_results, dict)
    assert set(raw_results) == set(["house_points", "quidditch_wins"])
    for measure in raw_results:
        result = raw_results[measure]
        assert isinstance(result, pd.DataFrame)
        assert (result[VALUE_COLUMN] == 0).all()
    STUDENT_HOUSES_LIST = list(STUDENT_HOUSES)

    # Check that indexes are as expected

    # Multi-stratification index is type MultiIndex where each layer dtype is Category
    expected_house_points_multi_idx = pd.MultiIndex.from_product(
        [POWER_LEVEL_GROUP_LABELS, STUDENT_HOUSES_LIST],
        names=["power_level", "student_house"],
    )
    # HACK: We need to set the levels to be CategoricalDtype but you can't set that
    # directly on the MultiIndex. Convert to df, set type, convert back
    expected_house_points_idx = (
        pd.DataFrame(index=expected_house_points_multi_idx)
        .reset_index()
        .astype(CategoricalDtype)
        .set_index(["power_level", "student_house"])
        .index
    )
    assert raw_results["house_points"].index.equals(expected_house_points_idx)

    # Single-stratification index is type CategoricalIndex
    expected_quidditch_idx = pd.CategoricalIndex(FAMILIARS, name="familiar")
    assert raw_results["quidditch_wins"].index.equals(expected_quidditch_idx)


def test_no_stratifications__raw_results_initialization() -> None:
    """Test that Observers requesting no stratifications result in a
    single-row DataFrame with 'value' of zero and index labeled 'all'
    """
    components = [Hogwarts(), NoStratificationsQuidditchWinsObserver()]
    sim = InteractiveContext(configuration=HARRY_POTTER_CONFIG, components=components)
    raw_results = sim._results._raw_results["no_stratifications_quidditch_wins"]
    assert isinstance(raw_results, pd.DataFrame)
    assert raw_results.shape == (1, 1)
    assert raw_results[VALUE_COLUMN].iat[0] == 0
    assert raw_results.index.equals(pd.Index(["all"]))


def test_observers_with_missing_stratifications_fail() -> None:
    """Test that an error is raised if an Observer requests a stratification
    that never actually gets registered.
    """
    components = [QuidditchWinsObserver(), HousePointsObserver(), Hogwarts()]

    expected_log_msg = re.escape(
        "The following stratifications are used by observers but not registered: \n"
        "['familiar', 'power_level_group', 'student_house']"
    )

    with pytest.raises(ValueError, match=expected_log_msg):
        InteractiveContext(configuration=HARRY_POTTER_CONFIG, components=components)


def test_unused_stratifications_are_logged(caplog: LogCaptureFixture) -> None:
    """Test that we issue a logger.info warning if Stratifications are registered
    but never actually used by an Observer

    The HogwartsResultsStratifier registers "student_house", "familiar", and
    "power_level_group" stratifiers. However, we will only use the QuidditchWinsObserver
    which only uses "familiar" and the MagicalAttributesObserver which only uses
    "power_level_group". We would thus expect only "student_house" to be logged
    as an unused stratification.

    """
    components = [
        Hogwarts(),
        HogwartsResultsStratifier(),
        QuidditchWinsObserver(),
        MagicalAttributesObserver(),
    ]
    InteractiveContext(configuration=HARRY_POTTER_CONFIG, components=components)

    log_split = caplog.text.split(
        "The following stratifications are registered but not used by any observers: \n"
    )
    # Check that the log message is present and only exists one time
    assert len(log_split) == 2
    # Check that the log message contains the expected Stratifications
    assert "['student_house']" in log_split[1]


def test_gather_results_with_no_observations(mocker: pytest_mock.MockerFixture) -> None:
    """Test that gather_results short-circuits when there are no observations for an event."""

    mgr = ResultsManager()
    mgr.population_view = mocker.Mock()
    mgr._results_context = mocker.Mock()
    mgr._results_context.get_observations.return_value = []  # type: ignore[attr-defined]

    event = Event(
        name=lifecycle_states.COLLECT_METRICS,
        index=pd.Index([0]),
        user_data={},
        time=0,
        step_size=1,
    )

    mgr.gather_results(event)

    mgr._results_context.get_observations.assert_called_once_with(event)  # type: ignore[attr-defined]
    mgr.population_view.subview.assert_not_called()  # type: ignore[attr-defined]
    mgr._results_context.gather_results.assert_not_called()  # type: ignore[attr-defined]


def test_gather_results_with_empty_index(mocker: pytest_mock.MockerFixture) -> None:
    """Test that gather_results short-circuits when an event has an empty index."""

    mgr = ResultsManager()
    mgr.population_view = mocker.Mock()
    mgr._results_context = mocker.Mock()
    mgr._results_context.get_observations.return_value = [mocker.Mock(spec=AddingObservation)]  # type: ignore[attr-defined]

    event = Event(
        name=lifecycle_states.COLLECT_METRICS,
        index=pd.Index([]),
        user_data={},
        time=0,
        step_size=1,
    )

    mgr.gather_results(event)

    mgr._results_context.get_observations.assert_called_once_with(event)  # type: ignore[attr-defined]
    mgr.population_view.subview.assert_not_called()  # type: ignore[attr-defined]
    mgr._results_context.gather_results.assert_not_called()  # type: ignore[attr-defined]


def test_gather_results_with_different_stratifications_and_to_observes() -> None:
    components = [
        Hogwarts(),
        HogwartsResultsStratifier(),
        NoStratificationsQuidditchWinsObserver(),
        NeverObserver(),
    ]
    sim = InteractiveContext(configuration=HARRY_POTTER_CONFIG, components=components)

    initial_raw_results = sim._results._raw_results.copy()

    sim.step()
    pd.testing.assert_frame_equal(
        sim._results._raw_results["never"], initial_raw_results["never"]
    )
    assert (
        sim._results._raw_results["no_stratifications_quidditch_wins"][VALUE_COLUMN] > 0
    ).all()


@pytest.fixture(scope="module")
def prepare_population_sim() -> InteractiveContext:
    return InteractiveContext(configuration=HARRY_POTTER_CONFIG, components=[Hogwarts()])


@pytest.mark.parametrize(
    "observation_requirements, stratification_requirements, expected_columns",
    [
        ([([], [])], [], []),
        ([([], [])], [(["power_level"], [])], ["power_level", "strat_0_mapped_values"]),
        ([(["familiar"], [])], [], ["familiar"]),
        (
            [(["familiar", "house_points"], [])],
            [(["power_level"], [])],
            ["familiar", "house_points", "power_level", "strat_0_mapped_values"],
        ),
        (
            [(["familiar"], []), (["house_points"], [])],
            [(["power_level"], [])],
            ["familiar", "house_points", "power_level", "strat_0_mapped_values"],
        ),
        (
            [(["familiar"], ["grade"])],
            [(["power_level"], ["double_power"])],
            ["familiar", "grade", "power_level", "double_power", "strat_0_mapped_values"],
        ),
        (
            [(["familiar"], [])],
            [(["power_level"], []), (["house_points"], [])],
            [
                "familiar",
                "power_level",
                "house_points",
                "strat_0_mapped_values",
                "strat_1_mapped_values",
            ],
        ),
        (
            [(["current_time", "event_step_size", "familiar"], [])],
            [(["event_time"], [])],
            [
                "current_time",
                "event_time",
                "event_step_size",
                "familiar",
                "strat_0_mapped_values",
            ],
        ),
        (
            [(["train"], [])],
            [(["headmaster"], [])],
            ["train", "headmaster", "strat_0_mapped_values"],
        ),
    ],
    ids=[
        "no_observation_requirements_no_stratifications",
        "no_observation_requirements",
        "no_stratifications",
        "multiple_columns_single_observation",
        "multiple_observations_with_columns",
        "columns_and_values",
        "multiple_stratifications",
        "time_data_and_columns",
        "user_data",
    ],
)
def test_prepare_population(
    prepare_population_sim: InteractiveContext,
    observation_requirements: list[tuple[list[str], list[str]]],
    stratification_requirements: list[tuple[list[str], list[str]]],
    expected_columns: list[str],
) -> None:
    mgr = prepare_population_sim._results
    observations: list[Observation] = [
        AddingObservation(
            name=f"test_observation_{i}",
            pop_filter="",
            when=lifecycle_states.COLLECT_METRICS,
            requires_columns=columns,
            requires_values=[prepare_population_sim.get_value(value) for value in values],
            results_formatter=lambda *_: pd.DataFrame(),
            aggregator_sources=[],
            aggregator=lambda *_: pd.Series(),
        )
        for i, (columns, values) in enumerate(observation_requirements)
    ]
    stratifications = [
        Stratification(
            name=f"strat_{i}",
            categories=["a", "b", "c"],
            excluded_categories=[],
            requires_columns=columns,
            requires_values=[prepare_population_sim.get_value(value) for value in values],
            mapper=lambda x: pd.Series("a", index=x.index),
            is_vectorized=True,
        )
        for i, (columns, values) in enumerate(stratification_requirements)
    ]

    event = Event(
        name=lifecycle_states.COLLECT_METRICS,
        index=prepare_population_sim.get_population().index,
        user_data={
            "train": "Hogwarts Express",
            "headmaster": "Albus Dumbledore",
            "country": "Scotland",
        },
        time=prepare_population_sim._clock.time + prepare_population_sim._clock.step_size,  # type: ignore [operator]
        step_size=prepare_population_sim._clock.step_size,
    )

    population = mgr._prepare_population(event, observations, stratifications)

    assert set(population.columns) == set(["tracked"] + expected_columns)
    if "current_time" in expected_columns:
        assert (population["current_time"] == prepare_population_sim._clock.time).all()
    if "event_time" in expected_columns:
        assert (population["event_time"] == event.time).all()
    if "event_step_size" in expected_columns:
        assert (population["event_step_size"] == event.step_size).all()
    if "train" in expected_columns:
        assert (population["train"] == "Hogwarts Express").all()
    for strat in stratifications:
        assert (
            population[get_mapped_col_name(strat.name)]
            == strat.stratify(population[strat._sources])
        ).all()


def test_stratified_observation_results() -> None:
    components = [
        Hogwarts(),
        CatBombObserver(),
        HogwartsResultsStratifier(),
    ]
    sim = InteractiveContext(configuration=HARRY_POTTER_CONFIG, components=components)
    assert (sim.get_results()["cat_bomb"]["value"] == 0.0).all()
    sim.step()
    num_familiars = sim.get_population().groupby(["familiar", "student_house"]).apply(len)
    expected = num_familiars.loc["cat"] ** 1.0
    expected.name = "value"
    expected = expected.sort_values().reset_index()
    expected["student_house"] = expected["student_house"].astype(
        CategoricalDtype(categories=STUDENT_HOUSES)
    )
    assert expected.equals(
        sim.get_results()["cat_bomb"].sort_values("value").reset_index(drop=True)
    )
    sim.step()
    num_familiars = sim.get_population().groupby(["familiar", "student_house"]).apply(len)
    expected = num_familiars.loc["cat"] ** 2.0
    expected.name = "value"
    expected = expected.sort_values().reset_index()
    expected["student_house"] = expected["student_house"].astype(
        CategoricalDtype(categories=STUDENT_HOUSES)
    )
    assert expected.equals(
        sim.get_results()["cat_bomb"].sort_values("value").reset_index(drop=True)
    )
    _assert_standard_index(sim.get_results()["cat_bomb"])


def test_unstratified_observation_results() -> None:
    components = [
        Hogwarts(),
        ValedictorianObserver(),
    ]
    sim = InteractiveContext(configuration=HARRY_POTTER_CONFIG, components=components)
    sim.step()
    first_valedictorian = sim.get_results()["valedictorian"]
    assert len(first_valedictorian) == 1
    sim.step()
    second_valedictorian = sim.get_results()["valedictorian"]
    assert len(second_valedictorian) == 1
    assert (
        first_valedictorian["student_id"].iat[0] != second_valedictorian["student_id"].iat[0]
    )
    _assert_standard_index(first_valedictorian)
    _assert_standard_index(second_valedictorian)


def test_concatenating_observation_results() -> None:
    components = [
        Hogwarts(),
        ExamScoreObserver(),
    ]
    sim = InteractiveContext(configuration=HARRY_POTTER_CONFIG, components=components)
    sim.step()
    results_one_step = sim.get_results()["exam_score"]
    assert (results_one_step["exam_score"] == 10.0).all()
    sim.step()
    expected_two_steps = results_one_step.copy()
    expected_two_steps["exam_score"] = 20.0
    expected_two_steps["event_time"] = results_one_step["event_time"] + pd.Timedelta(
        days=sim.configuration.time.step_size
    )
    assert sim.get_results()["exam_score"].equals(
        pd.concat([results_one_step, expected_two_steps], axis=0).reset_index(drop=True)
    )
    _assert_standard_index(sim.get_results()["exam_score"])


def test_adding_observation_results() -> None:
    """Test that adding observation results are being updated correctly."""

    def _check_house_points(pop: pd.DataFrame, step_number: int) -> None:
        """We know that house points are stratified by 'student_house' and 'power_level_group'.
        and that each wizard of gryffindor and of level 20 or 80 (which correspond
        to 'low' and 'very high' power level groups) gains a point
        """
        assert set(pop["house_points"]) == set([0, 1])
        assert (pop.loc[pop["house_points"] != 0, "student_house"] == "gryffindor").all()
        assert set(pop.loc[pop["house_points"] != 0, "power_level"]) == set([20, 80])
        group_sizes = pd.DataFrame(
            pop.groupby(["power_level", "student_house"]).size().astype("float"),
            columns=[VALUE_COLUMN],
        )
        raw_results = sim._results._raw_results["house_points"]
        # We cannot use `equals` here because raw results have a MultiIndex where
        # each layer is a Category dtype but pop has object dtype for the relevant columns
        assert (  # type: ignore[union-attr]
            raw_results.loc[(["low", "very high"], "gryffindor"), "value"].values
            == (group_sizes.loc[([20, 80], "gryffindor"), "value"] * step_number).values
        ).all()

    def _check_quidditch_wins(pop: pd.DataFrame, step_number: int) -> None:
        """We know that quidditch wins are stratified by 'familiar'
        and that each wizard with a banana slug familiar gains a point
        """
        assert set(pop["quidditch_wins"]) == set([0, 1])
        assert (pop.loc[pop["quidditch_wins"] != 0, "familiar"] == "banana_slug").all()
        group_sizes = pd.DataFrame(
            pop.groupby(["familiar"]).size().astype("float"), columns=[VALUE_COLUMN]
        )
        raw_results = sim._results._raw_results["quidditch_wins"]
        # We cannot use `equals` here because raw results have a MultiIndex where
        # each layer is a Category dtype but pop has object dtype for the relevant columns
        assert (
            raw_results[raw_results[VALUE_COLUMN] != 0].values
            == (group_sizes[group_sizes.index == "banana_slug"] * step_number).values
        ).all()

    components = [
        Hogwarts(),
        HousePointsObserver(),
        QuidditchWinsObserver(),
        HogwartsResultsStratifier(),
    ]
    sim = InteractiveContext(configuration=HARRY_POTTER_CONFIG, components=components)
    sim.step()
    pop = sim.get_population()
    _check_house_points(pop, step_number=1)
    _check_quidditch_wins(pop, step_number=1)

    sim.step()
    pop = sim.get_population()
    _check_house_points(pop, step_number=2)
    _check_quidditch_wins(pop, step_number=2)
    _assert_standard_index(sim.get_results()["house_points"])
    _assert_standard_index(sim.get_results()["quidditch_wins"])


def test_concatenating_observation_updates() -> None:
    """Test that concatenating observation raw results are being updated correctly."""
    components = [
        Hogwarts(),
        ExamScoreObserver(),
    ]
    sim = InteractiveContext(configuration=HARRY_POTTER_CONFIG, components=components)
    sim.step()
    results_one_step = sim.get_results()["exam_score"]
    assert (results_one_step["exam_score"] == 10.0).all()
    sim.step()
    expected_two_steps = results_one_step.copy()
    expected_two_steps["exam_score"] = 20.0
    expected_two_steps["event_time"] = results_one_step["event_time"] + pd.Timedelta(
        days=sim.configuration.time.step_size
    )
    assert sim.get_results()["exam_score"].equals(
        pd.concat([results_one_step, expected_two_steps], axis=0).reset_index(drop=True)
    )


def test_update__raw_results_fully_filtered_pop() -> None:
    components = [
        Hogwarts(),
        FullyFilteredHousePointsObserver(),
        HogwartsResultsStratifier(),
    ]
    sim = InteractiveContext(configuration=HARRY_POTTER_CONFIG, components=components)
    sim.step()
    # The FullyFilteredHousePointsObserver filters the population to a bogus
    # power level and so we should not be observing anything
    assert (sim._results._raw_results["house_points"][VALUE_COLUMN] == 0).all()
    sim.step()
    assert (sim._results._raw_results["house_points"][VALUE_COLUMN] == 0).all()


def test_update__raw_results_no_stratifications() -> None:
    components = [Hogwarts(), NoStratificationsQuidditchWinsObserver()]
    sim = InteractiveContext(configuration=HARRY_POTTER_CONFIG, components=components)
    sim.step()
    pop = sim.get_population()
    raw_results = sim._results._raw_results["no_stratifications_quidditch_wins"]
    assert raw_results.loc["all"][VALUE_COLUMN] == pop["quidditch_wins"].sum()
    sim.step()
    pop = sim.get_population()
    raw_results = sim._results._raw_results["no_stratifications_quidditch_wins"]
    assert raw_results.loc["all"][VALUE_COLUMN] == pop["quidditch_wins"].sum() * 2


def test_update__raw_results_extra_columns() -> None:
    """Test that raw results are updated correctly when the aggregator return
    contains multiple columns (i.e. not just a single 'value' column)
    """
    components = [Hogwarts(), HogwartsResultsStratifier(), MagicalAttributesObserver()]
    sim = InteractiveContext(configuration=HARRY_POTTER_CONFIG, components=components)
    sim.step()
    raw_results = sim._results._raw_results["magical_attributes"]
    assert (raw_results[["spell_power", "potion_power"]].values == [1, 1]).all()
    sim.step()
    raw_results = sim._results._raw_results["magical_attributes"]
    assert (raw_results[["spell_power", "potion_power"]].values == [2, 2]).all()


####################
# Helper functions #
####################


def _assert_standard_index(df: pd.DataFrame) -> None:
    """The results should have any stratifications as columns, not indexes."""
    assert not isinstance(df.index, pd.MultiIndex)
    assert df.index.names == [None]
    assert df.index.dtype == "int64"
