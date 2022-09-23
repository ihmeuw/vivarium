from types import MethodType

import pytest

from vivarium.framework.results.manager import ResultsManager

from .mocks import CATEGORIES, NAME, SOURCES, sorting_hat_serial, sorting_hat_vector


# Mock for get_value call for Pipelines, returns a str instead of a Pipeline
def mock_get_value(self, name: str):
    return name


def verify_stratification_added(mgr, name, categories, sources, mapper, is_vectorized):
    matching_stratification_found = False
    for stratification in mgr._results_context._stratifications:  # noqa
        # big equality check
        if (
            stratification.name == name
            and stratification.categories == categories
            and stratification.mapper == mapper
            and stratification.is_vectorized == is_vectorized
            and stratification.sources == sources
        ):
            matching_stratification_found = True
        break
    assert matching_stratification_found


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
    verify_stratification_added(mgr, name, categories, sources, mapper, is_vectorized)


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
    verify_stratification_added(mgr, name, categories, sources, mapper, is_vectorized)


# def test_duplicate_register_stratification(mocker):
#     mgr = ResultsManager()
#     builder = mocker.Mock()
#     mgr.setup(builder)
#     mgr.register_stratification(NAME, CATEGORIES, sorting_hat_serial, False, SOURCES, [])
#     mgr.register_stratification(NAME, CATEGORIES, sorting_hat_serial, False, SOURCES, [])
# TODO: Decide what should happen on duplicate stratifications:
#   a) context throws a ValueError, b) overwrite the existing named Stratification, c) keep dupes?
