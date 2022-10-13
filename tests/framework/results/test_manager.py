from types import MethodType

import pytest

from vivarium.framework.results.manager import ResultsManager

from .mocks import (
    BIN_BINNED_COLUMN,
    BIN_LABELS,
    BIN_SILLY_BINS,
    BIN_SOURCE,
    CATEGORIES,
    NAME,
    SOURCES,
    sorting_hat_serial,
    sorting_hat_vector,
    verify_stratification_added,
)


# Mock for get_value call for Pipelines, returns a str instead of a Pipeline
def mock_get_value(self, name: str):
    return name


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
        mgr._results_context._stratifications,
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
        mgr._results_context._stratifications,
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
        mgr._results_context._stratifications,
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
