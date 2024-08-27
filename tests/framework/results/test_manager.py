import re
from types import MethodType

import numpy as np
import pandas as pd
import pytest
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
    NoStratificationsQuidditchWinsObserver,
    QuidditchWinsObserver,
    ValedictorianObserver,
    mock_get_value,
    sorting_hat_serial,
    sorting_hat_vectorized,
    verify_stratification_added,
)
from vivarium.framework.results import VALUE_COLUMN
from vivarium.framework.results.context import ResultsContext
from vivarium.framework.results.manager import ResultsManager
from vivarium.framework.results.observation import AddingObservation
from vivarium.interface.interactive import InteractiveContext


@pytest.mark.parametrize(
    "stratifications, default_stratifications, additional_stratifications, excluded_stratifications, expected_stratifications",
    [
        ([], [], [], [], ()),
        (
            [],
            ["age", "sex"],
            ["handedness"],
            ["age"],
            ("sex", "handedness"),
        ),
        ([], ["age", "sex"], [], ["age", "sex"], ()),
        ([], ["age"], [], ["bogus_exclude_column"], ("age",)),
        (["custom"], ["age", "sex"], [], [], ("custom", "age", "sex")),
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
    stratifications,
    default_stratifications,
    additional_stratifications,
    excluded_stratifications,
    expected_stratifications,
    mocker,
):
    ctx = ResultsContext()
    ctx.default_stratifications = default_stratifications
    mgr = ResultsManager()
    mocker.patch.object(mgr, "_results_context", ctx)
    # default_stratifications would normally be set via ResultsInterface.set_default_stratifications()
    stratifications = mgr._get_stratifications(
        stratifications, additional_stratifications, excluded_stratifications
    )
    assert sorted(stratifications) == sorted(expected_stratifications)


#######################################
# Tests for `register_stratification` #
#######################################


@pytest.mark.parametrize(
    "name, sources, categories, excluded_categories, mapper, is_vectorized",
    [
        (
            NAME,
            NAME_COLUMNS,
            HOUSE_CATEGORIES,
            [],
            sorting_hat_vectorized,
            True,
        ),
        (
            NAME,
            NAME_COLUMNS,
            HOUSE_CATEGORIES,
            [],
            sorting_hat_serial,
            False,
        ),
        (
            NAME,
            NAME_COLUMNS,
            HOUSE_CATEGORIES,
            ["gryffindor"],
            sorting_hat_vectorized,
            True,
        ),
    ],
    ids=["vectorized_mapper", "non-vectorized_mapper", "excluded_categories"],
)
def test_register_stratification_no_pipelines(
    name, sources, categories, excluded_categories, mapper, is_vectorized, mocker
):
    mgr = ResultsManager()
    builder = mocker.Mock()
    builder.configuration.stratification = LayeredConfigTree(
        {"default": [], "excluded_categories": {}}
    )
    mgr.setup(builder)
    mgr.register_stratification(
        name=name,
        categories=categories,
        excluded_categories=excluded_categories,
        mapper=mapper,
        is_vectorized=is_vectorized,
        requires_columns=sources,
    )
    for item in sources:
        assert item in mgr._required_columns
    assert verify_stratification_added(
        mgr._results_context.stratifications,
        name,
        sources,
        categories,
        excluded_categories,
        mapper,
        is_vectorized,
    )


@pytest.mark.parametrize(
    "name, sources, categories, mapper, is_vectorized",
    [
        (
            NAME,
            NAME_COLUMNS,
            HOUSE_CATEGORIES,
            sorting_hat_vectorized,
            True,
        ),
        (
            NAME,
            NAME_COLUMNS,
            HOUSE_CATEGORIES,
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
    builder.configuration.stratification = LayeredConfigTree(
        {"default": [], "excluded_categories": {}}
    )
    # Set up mock builder with mocked get_value call for Pipelines
    mocker.patch.object(builder, "value.get_value")
    builder.value.get_value = MethodType(mock_get_value, builder)
    mgr.setup(builder)
    mgr.register_stratification(
        name=name,
        categories=categories,
        excluded_categories=None,
        mapper=mapper,
        is_vectorized=is_vectorized,
        requires_columns=[],
        requires_values=sources,
    )
    for item in sources:
        assert item in mgr._required_values
    assert verify_stratification_added(
        mgr._results_context.stratifications,
        name,
        sources,
        categories,
        [],
        mapper,
        is_vectorized,
    )


@pytest.mark.parametrize(
    "name, sources, categories, mapper, is_vectorized",
    [
        (  # expected Stratification for vectorized
            NAME,
            NAME_COLUMNS,
            HOUSE_CATEGORIES,
            sorting_hat_vectorized,
            True,
        ),
        (  # expected Stratification for non-vectorized
            NAME,
            NAME_COLUMNS,
            HOUSE_CATEGORIES,
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
    builder.configuration.stratification = LayeredConfigTree(
        {"default": [], "excluded_categories": {}}
    )
    # Set up mock builder with mocked get_value call for Pipelines
    mocker.patch.object(builder, "value.get_value")
    builder.value.get_value = MethodType(mock_get_value, builder)
    mgr.setup(builder)
    mocked_column_name = "silly_column"
    mgr.register_stratification(
        name=name,
        categories=categories,
        excluded_categories=None,
        mapper=mapper,
        is_vectorized=is_vectorized,
        requires_columns=[mocked_column_name],
        requires_values=sources,
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
        [],
        mapper,
        is_vectorized,
    )


##############################################
# Tests for `register_binned_stratification` #
##############################################


@pytest.mark.parametrize(
    "bins, labels",
    [(BIN_SILLY_BIN_EDGES, BIN_LABELS[1:]), (BIN_SILLY_BIN_EDGES[1:], BIN_LABELS)],
    ids=["too_many_bins", "too_many_labels"],
)
def test_register_binned_stratification_raises_bins_labels_mismatch(bins, labels):
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


def test_binned_stratification_mapper():
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
    strat = mgr._results_context.stratifications[0]
    data = pd.Series([-np.inf] + BIN_SILLY_BIN_EDGES + [np.inf])
    groups = strat.mapper(data)
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
    default, additional, excluded, match, mocker, caplog
):
    mgr = ResultsManager()
    builder = mocker.Mock()
    mgr.setup(builder)
    mgr.logger = logger

    mgr._results_context.default_stratifications = default
    mgr.register_observation(
        observation_type=AddingObservation,
        is_stratified=True,
        name="name",
        pop_filter='alive == "alive"',
        aggregator_sources=[],
        aggregator=lambda: None,
        requires_columns=[],
        requires_values=[],
        additional_stratifications=additional,
        excluded_stratifications=excluded,
        when="collect_metrics",
        results_formatter=lambda: None,
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


def test__raw_results_initialized_as_empty_dict(mocker):
    """Test that raw results are initialized as an empty dictionary"""
    mgr = ResultsManager()
    builder = mocker.Mock()
    mgr.setup(builder)
    assert mgr._raw_results == {}


def test_stratified__raw_results_initialization():
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
    expected_house_points_idx = pd.MultiIndex.from_product(
        [STUDENT_HOUSES_LIST, POWER_LEVEL_GROUP_LABELS],
        names=["student_house", "power_level"],
    )
    # HACK: We need to set the levels to be CategoricalDtype but you can't set that
    # directly on the MultiIndex. Convert to df, set type, convert back
    expected_house_points_idx = (
        pd.DataFrame(index=expected_house_points_idx)
        .reset_index()
        .astype(CategoricalDtype)
        .set_index(["student_house", "power_level"])
        .index
    )
    assert raw_results["house_points"].index.equals(expected_house_points_idx)

    # Single-stratification index is type CategoricalIndex
    expected_quidditch_idx = pd.CategoricalIndex(FAMILIARS, name="familiar")
    assert raw_results["quidditch_wins"].index.equals(expected_quidditch_idx)


def test_no_stratifications__raw_results_initialization():
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


def test_observers_with_missing_stratifications_fail():
    """Test that an error is raised if an Observer requests a stratification
    that never actually gets registered.
    """
    components = [QuidditchWinsObserver(), HousePointsObserver(), Hogwarts()]

    expected_missing = ["familiar", "power_level_group", "student_house"]
    expected_log_msg = re.escape(
        "The following observers are requested to be stratified by stratifications "
        f"that are not registered: \n{expected_missing}"
    )

    with pytest.raises(ValueError, match=expected_log_msg):
        InteractiveContext(configuration=HARRY_POTTER_CONFIG, components=components)


def test_unused_stratifications_are_logged(caplog):
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


def test_stratified_observation_results():
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
    assert expected.sort_values().equals(sim.get_results()["cat_bomb"]["value"].sort_values())
    sim.step()
    num_familiars = sim.get_population().groupby(["familiar", "student_house"]).apply(len)
    expected = num_familiars.loc["cat"] ** 2.0
    expected.name = "value"
    assert expected.sort_values().equals(sim.get_results()["cat_bomb"]["value"].sort_values())


def test_unstratified_observation_results():
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


def test_concatenating_observation_results():
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


def test_adding_observation_results():
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
            pop.groupby(["student_house", "power_level"]).size().astype("float"),
            columns=[VALUE_COLUMN],
        )
        raw_results = sim._results._raw_results["house_points"]
        # We cannot use `equals` here because raw results have a MultiIndex where
        # each layer is a Category dtype but pop has object dtype for the relevant columns
        assert (
            raw_results.loc[("gryffindor", ["low", "very high"]), "value"].values
            == (group_sizes.loc[("gryffindor", [20, 80]), "value"] * step_number).values
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


def test_concatenating_observation_updates():
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


def test_update__raw_results_fully_filtered_pop():
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


def test_update__raw_results_no_stratifications():
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


def test_update__raw_results_extra_columns():
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
