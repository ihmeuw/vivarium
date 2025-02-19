import itertools
from typing import Sequence

import numpy as np
import pandas as pd
import pytest
from layered_config_tree import LayeredConfigTree
from pytest_mock import MockerFixture

from vivarium import InteractiveContext
from vivarium.framework.lookup import (
    LookupTableInterface,
    LookupTableManager,
    validate_build_table_parameters,
)
from vivarium.framework.lookup.table import InterpolatedTable
from vivarium.testing_utilities import TestPopulation, build_table
from vivarium.types import LookupTableData


@pytest.mark.skip(reason="only order 0 interpolation currently supported")
def test_interpolated_tables(base_config: LayeredConfigTree) -> None:
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    years_df = build_table(
        lambda x: x[0],
        parameter_columns={
            "year": (year_start, year_end),
            "age": (0, 125),
        },
    )
    ages_df = build_table(
        lambda x: x[0],
        parameter_columns={
            "year": (year_start, year_end),
            "age": (0, 125),
        },
    )
    one_d_age_df = ages_df.copy().drop_duplicates()
    base_config.update(
        {"population": {"population_size": 10000}, "interpolation": {"order": 1}}
    )  # the results we're checking later assume interp order 1

    simulation = InteractiveContext(components=[TestPopulation()], configuration=base_config)
    manager = simulation._tables
    years = manager.build_table(
        years_df,
        key_columns=("sex",),
        parameter_columns=(
            "age",
            "year",
        ),
        value_columns=(),
    )
    ages = manager.build_table(
        ages_df,
        key_columns=("sex",),
        parameter_columns=(
            "age",
            "year",
        ),
        value_columns=(),
    )
    one_d_age = manager.build_table(
        one_d_age_df, key_columns=("sex",), parameter_columns=("age",), value_columns=()
    )

    pop = simulation.get_population(untracked=True)
    result_years = years(pop.index)
    result_ages = ages(pop.index)
    result_ages_1d = one_d_age(pop.index)

    fractional_year = simulation._clock.time.year  # type: ignore [union-attr]
    fractional_year += simulation._clock.time.timetuple().tm_yday / 365.25  # type: ignore [union-attr]

    assert np.allclose(result_years, fractional_year)
    assert np.allclose(result_ages, pop.age)
    assert np.allclose(result_ages_1d, pop.age)

    simulation._clock._clock_time += pd.Timedelta(30.5 * 125, unit="D")  # type: ignore [operator]
    simulation._population._population.age += 125 / 12  # type: ignore [union-attr]

    result_years = years(pop.index)
    result_ages = ages(pop.index)
    result_ages_1d = one_d_age(pop.index)

    fractional_year = simulation._clock.time.year  # type: ignore [union-attr]
    fractional_year += simulation._clock.time.timetuple().tm_yday / 365.25  # type: ignore [union-attr]

    assert np.allclose(result_years, fractional_year)
    assert np.allclose(result_ages, pop.age)
    assert np.allclose(result_ages_1d, pop.age)


@pytest.mark.skip(reason="only order 0 interpolation currently supported")
def test_interpolated_tables_without_uninterpolated_columns(
    base_config: LayeredConfigTree,
) -> None:
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    years_df = build_table(
        lambda x: x[0],
        parameter_columns={
            "year": (year_start, year_end),
            "age": (0, 125),
        },
    )
    del years_df["sex"]
    years_df = years_df.drop_duplicates()
    base_config.update(
        {"population": {"population_size": 10000}, "interpolation": {"order": 1}}
    )  # the results we're checking later assume interp order 1

    simulation = InteractiveContext(components=[TestPopulation()], configuration=base_config)
    manager = simulation._tables
    years = manager.build_table(
        years_df,
        key_columns=(),
        parameter_columns=(
            "year",
            "age",
        ),
        value_columns=(),
    )

    result_years = years(simulation.get_population().index)

    fractional_year = simulation._clock.time.year  # type: ignore [union-attr]
    fractional_year += simulation._clock.time.timetuple().tm_yday / 365.25  # type: ignore [union-attr]

    assert np.allclose(result_years, fractional_year)

    simulation._clock._clock_time += pd.Timedelta(30.5 * 125, unit="D")  # type: ignore [operator]

    result_years = years(simulation.get_population().index)

    fractional_year = simulation._clock.time.year  # type: ignore [union-attr]
    fractional_year += simulation._clock.time.timetuple().tm_yday / 365.25  # type: ignore [union-attr]

    assert np.allclose(result_years, fractional_year)


def test_interpolated_tables__exact_values_at_input_points(
    base_config: LayeredConfigTree,
) -> None:
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    years_df = build_table(
        lambda x: x[0],
        parameter_columns={
            "year": (year_start, year_end),
            "age": (0, 125),
        },
    )

    input_years = years_df.year_start.unique()
    base_config.update({"population": {"population_size": 10000}})

    simulation = InteractiveContext(components=[TestPopulation()], configuration=base_config)
    manager = simulation._tables
    years = manager._build_table(
        years_df, key_columns=["sex"], parameter_columns=["age", "year"], value_columns=()
    )

    for year in input_years:
        simulation._clock._clock_time = pd.Timestamp(year, 1, 1)
        assert np.allclose(
            years(simulation.get_population().index), simulation._clock.time.year + 1 / 365  # type: ignore [union-attr]
        )


def test_interpolated_tables__only_categorical_parameters(
    base_config: LayeredConfigTree,
) -> None:
    sexes = ["Female", "Male"]
    locations = ["USA", "Canada", "Mexico"]
    combinations = enumerate(itertools.product(sexes, locations))
    input_data = pd.DataFrame(
        [
            {"sex": sex, "location": location, "some_value": i**2}
            for i, (sex, location) in combinations
        ]
    )

    base_config.update({"population": {"population_size": 10000}})

    simulation = InteractiveContext(components=[TestPopulation()], configuration=base_config)
    manager = simulation._tables
    lookup_table = manager._build_table(
        input_data, key_columns=["sex", "location"], parameter_columns=(), value_columns=()
    )

    population = simulation.get_population()[["sex", "location"]]
    output_data = lookup_table(population.index)

    for i, (sex, location) in combinations:
        sub_table_mask = (output_data["sex"] == sex) & output_data["location"] == location
        assert (output_data.loc[sub_table_mask, "some_value"] == i**2).all()


@pytest.mark.parametrize("data", [(1, 2), [1, 2]])
def test_lookup_table_scalar_from_list(
    base_config: LayeredConfigTree, data: Sequence[int]
) -> None:
    simulation = InteractiveContext(components=[TestPopulation()], configuration=base_config)
    manager = simulation._tables
    table = manager._build_table(
        data, key_columns=(), parameter_columns=(), value_columns=["a", "b"]  # type: ignore [arg-type]
    )(simulation.get_population().index)

    assert isinstance(table, pd.DataFrame)
    assert table.columns.values.tolist() == ["a", "b"]
    assert np.all(table.a == 1)
    assert np.all(table.b == 2)


def test_lookup_table_scalar_from_single_value(base_config: LayeredConfigTree) -> None:
    simulation = InteractiveContext(components=[TestPopulation()], configuration=base_config)
    manager = simulation._tables
    table = manager._build_table(
        1, key_columns=(), parameter_columns=(), value_columns=["a"]
    )(simulation.get_population().index)
    assert isinstance(table, pd.Series)
    assert np.all(table == 1)


def test_invalid_data_type_build_table(base_config: LayeredConfigTree) -> None:
    simulation = InteractiveContext(components=[TestPopulation()], configuration=base_config)
    manager = simulation._tables
    with pytest.raises(TypeError):
        manager._build_table("break", key_columns=(), parameter_columns=(), value_columns=())  # type: ignore [arg-type]


def test_lookup_table_interpolated_return_types(base_config: LayeredConfigTree) -> None:
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    data = build_table(
        lambda x: x[0],
        parameter_columns={
            "year": (year_start, year_end),
            "age": (0, 125),
        },
    )

    simulation = InteractiveContext(components=[TestPopulation()], configuration=base_config)
    manager = simulation._tables
    table = manager._build_table(
        data, key_columns=["sex"], parameter_columns=["age", "year"], value_columns=()
    )(simulation.get_population().index)
    # make sure a single value column is returned as a series
    assert isinstance(table, pd.Series)

    # now add a second value column to make sure the result is a df
    data["value2"] = data.value
    table = manager._build_table(
        data, key_columns=["sex"], parameter_columns=["age", "year"], value_columns=()
    )(simulation.get_population().index)

    assert isinstance(table, pd.DataFrame)


@pytest.mark.parametrize(
    "data", [None, pd.DataFrame(), pd.DataFrame(columns=["a", "b", "c"]), [], tuple()]
)
def test_validate_parameters_no_data(data: LookupTableData) -> None:
    with pytest.raises(ValueError, match="supply some data"):
        validate_build_table_parameters(data, [], [], [])


@pytest.mark.parametrize(
    "key_cols, param_cols, val_cols, match",
    [
        ((), (), (), "supply value_columns"),
        ((), (), [], "supply value_columns"),
        ((), (), ["a", "b"], "match the number of values"),
        (("a", "b"), (), ["d", "e", "f"], "key_columns are not allowed"),
        ((), ("a", "b"), ["d", "e", "f"], "parameter_columns are not allowed"),
    ],
)
def test_validate_parameters_error_scalar_data(
    key_cols: Sequence[str], param_cols: Sequence[str], val_cols: Sequence[str], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_build_table_parameters([1, 2, 3], key_cols, param_cols, val_cols)


@pytest.mark.parametrize("data", ["FAIL", pd.Interval(5, 10), "2019-05-17"])
def test_validate_parameters_fail_other_data(data: LookupTableData) -> None:
    with pytest.raises(TypeError, match="only allowable types"):
        validate_build_table_parameters(data, [], [], [])


@pytest.mark.parametrize(
    "key_cols, param_cols, val_cols, match",
    [
        ([], [], ["c"], "either key_columns or parameter_columns"),
        (["a", "b"], ["b"], ["c"], "no overlap between key.*and parameter columns"),
        (["a"], ["b"], ["a", "c"], "no overlap between value.*and key.*columns"),
        (["a"], ["b"], ["b", "c"], "no overlap between value.*and.*parameter columns"),
        (["d"], ["b"], ["c"], "columns.*must all be present"),
        (["a"], ["d"], ["c"], "columns.*must all be present"),
        (["a"], ["b"], ["d"], "columns.*must all be present"),
    ],
)
def test_validate_parameters_error_dataframe(
    key_cols: Sequence[str], param_cols: Sequence[str], val_cols: Sequence[str], match: str
) -> None:
    data = pd.DataFrame({"a": [1, 2], "b_start": [0, 5], "b_end": [5, 10], "c": [100, 150]})
    with pytest.raises(ValueError, match=match):
        validate_build_table_parameters(data, key_cols, param_cols, val_cols)


def test_validate_parameters_pass_scalar_data() -> None:
    validate_build_table_parameters([1, 2, 3], (), (), ["a", "b", "c"])


@pytest.mark.parametrize(
    "key_cols, param_cols, val_cols",
    [
        (["a"], ["b"], ["c"]),
        ([], ["b"], ["c", "a"]),
        ([], ["b"], ["a", "c"]),
        ([], ["b"], ["c"]),
        (["a"], [], ["c"]),
        (["a"], ["b"], []),
        (["a"], [], []),
        ([], ["b"], []),
    ],
)
def test_validate_parameters_pass_dataframe(
    key_cols: Sequence[str], param_cols: Sequence[str], val_cols: Sequence[str]
) -> None:
    data = pd.DataFrame({"a": [1, 2], "b_start": [0, 5], "b_end": [5, 10], "c": [100, 150]})
    validate_build_table_parameters(data, key_cols, param_cols, val_cols)


@pytest.mark.parametrize("validate", [True, False])
def test_validate_flag(mocker: MockerFixture, validate: bool) -> None:
    manager = LookupTableManager()
    manager.setup(mocker.Mock())
    manager._validate = validate
    interface = LookupTableInterface(manager)

    mock_validator = mocker.patch(
        "vivarium.framework.lookup.manager.validate_build_table_parameters"
    )

    interface.build_table(0, value_columns=["a"])

    if validate:
        mock_validator.assert_called_once()
    else:
        mock_validator.assert_not_called()


def test__build_table_from_dict(base_config: LayeredConfigTree) -> None:
    simulation = InteractiveContext(components=[TestPopulation()], configuration=base_config)
    manager = simulation._tables
    data = {
        "a_start": [0.0, 0.5, 1.0, 1.5],
        "a_end": [0.5, 1.0, 1.5, 2.0],
        "b": [10.0, 20.0, 30.0, 40.0],
        "c": [100.0, 200.0, 300.0, 400.0],
    }
    # We convert the dict to a dataframe before we call validate_build_table_parameters so
    # this test is really going to just ensure we don't error out when we pass in a dict and
    # we get the expected return type from _build_table
    table = manager._build_table(
        data,  # type: ignore [arg-type]
        key_columns=["b"],
        parameter_columns=["a"],
        value_columns=["c"],
    )
    assert isinstance(table, InterpolatedTable)
    assert table.key_columns == ["b"]
    assert table.parameter_columns == ["a"]
    assert table.value_columns == ["c"]
