from types import MethodType

import pandas as pd
import pytest
from loguru import logger

from vivarium.framework.results.manager import ResultsManager
from vivarium.interface.interactive import InteractiveContext

from ...helper_components import (
    CatActivityObserver,
    CatResultsStratifier,
    CatToyObserver,
)
from .mocks import (
    BIN_BINNED_COLUMN,
    BIN_LABELS,
    BIN_SILLY_BINS,
    BIN_SOURCE,
    CATEGORIES,
    NAME,
    SOURCES,
    mock_get_value,
    sorting_hat_serial,
    sorting_hat_vector,
    verify_stratification_added,
)

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
        "name",
        'alive == "alive"',
        [],
        lambda: None,
        additional_stratifications=additional,
        excluded_stratifications=excluded,
        when="collect_metrics",
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


def test_metrics_initialized_as_empty_dict(mocker):
    """Test that metrics are initialized as an empty dictionary"""
    mgr = ResultsManager()
    builder = mocker.Mock()
    mgr.setup(builder)
    assert mgr.metrics == {}


def test_stratified_metrics_initialized_as_zeros_dataframes():
    """Test that matrics are being initialized correctly. We expect a dictionary
    of pd.DataFrames. Each key of the dictionary is an observed measure name and
    the corresponding value is a pd.DataFrame with a multi-index of that observer's
    stratifications and an all-zeros 'value' column
    """

    components = [
        CatToyObserver(),
        CatActivityObserver(),
        CatResultsStratifier(),
    ]
    config = {
        "stratification": {
            "default": ["personality", "favorite_toy"],
        },
    }
    sim = InteractiveContext(configuration=config, components=components)
    metrics = sim._results.metrics
    assert isinstance(metrics, dict)
    assert set(metrics) == set(["cat_toy", "cat_activity"])
    for metric in metrics:
        result = metrics[metric]
        assert isinstance(result, pd.DataFrame)
        assert result.columns == ["value"]
        assert result["value"].unique() == [0.0]
    assert metrics["cat_toy"].index.equals(
        pd.MultiIndex.from_product(
            [["psycopath", "cantankerous"], ["string", "human_face"]],
            names=["personality", "favorite_toy"],
        )
    )
    assert metrics["cat_activity"].index.equals(
        pd.MultiIndex.from_product(
            [["psycopath", "cantankerous"], ["sleep", "eat"]],
            names=["personality", "favorite_activity"],
        )
    )
