import itertools

import pandas as pd
import pytest

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
from vivarium.framework.results.context import ResultsContext, SummingObservation


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
def test_summing_observation__aggregate(stratifications, aggregator_sources, aggregator):
    """Test that we are aggregating correctly. There are some nuances here:
    - If aggregator_resources is provided, then simply .apply it to the groups passed in.
    - If no aggregator_resources are provided, then we want a full aggregation of the groups.
    - _aggregate can return either a pd.Series or a pd.DataFrame of any number of columns
    """
    groups = ResultsContext()._get_groups(
        stratifications=stratifications, filtered_pop=BASE_POPULATION
    )
    obs = SummingObservation(
        name="foo",
        pop_filter="",
        when="whenevs",
        formatter=lambda _, __: pd.DataFrame(),
        stratifications=stratifications,
        aggregator_sources=aggregator_sources,
        aggregator=aggregator,
    )
    aggregates = obs._aggregate(
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
def test_summing_observation__format(aggregates):
    obs = SummingObservation(
        name="foo",
        pop_filter="",
        when="whenevs",
        formatter=lambda _, __: pd.DataFrame(),
        stratifications=(),
        aggregator_sources=None,
        aggregator=lambda _: 0.0,
    )
    new_aggregates = obs._format(aggregates=aggregates)
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
def test_summing_observation__expand_index(aggregates):
    obs = SummingObservation(
        name="foo",
        pop_filter="",
        when="whenevs",
        formatter=lambda _, __: pd.DataFrame(),
        stratifications=(),
        aggregator_sources=None,
        aggregator=lambda _: 0.0,
    )
    full_idx_aggregates = obs._expand_index(aggregates=aggregates)
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
