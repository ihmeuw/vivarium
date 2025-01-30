from __future__ import annotations

import random
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from layered_config_tree import LayeredConfigTree
from pytest_mock import MockerFixture

from vivarium.framework.artifact.manager import (
    ArtifactManager,
    _config_filter,
    _subset_columns,
    _subset_rows,
    parse_artifact_path_config,
    validate_filter_term,
)
from vivarium.testing_utilities import build_table, metadata


@pytest.fixture()
def artifact_mock(mocker: MockerFixture) -> MagicMock:
    mock = mocker.patch("vivarium.framework.artifact.manager.Artifact")

    def mock_load(key: str) -> str | pd.DataFrame | None:
        if key == "string_data.key":
            return "string_data"
        elif key == "df_data.key":
            return pd.DataFrame()
        else:
            return None

    mock.load.side_effect = mock_load

    return mock


def test_subset_rows_extra_filters() -> None:
    data = build_table(1)
    with pytest.raises(ValueError):
        _subset_rows(data, missing_thing=12)


def test_subset_rows() -> None:
    values = [
        lambda *args, **kwargs: random.choice(["red", "blue"]),
        lambda *args, **kwargs: random.choice([1, 2, 3]),
    ]
    data = build_table(
        value=values,
        value_columns=["color", "number"],
    )
    filtered_data = _subset_rows(data, color="red", number=3)
    assert filtered_data.equals(data[(data.color == "red") & (data.number == 3)])

    filtered_data = _subset_rows(data, color="red", number=[2, 3])
    assert filtered_data.equals(
        data[(data.color == "red") & ((data.number == 2) | (data.number == 3))]
    )


def test_subset_columns() -> None:
    values = [0, "red", 100]
    data = build_table(
        values,
        parameter_columns={
            "age": (0, 99),
            "year": (1990, 2010),
        },
        value_columns=["draw", "color", "value"],
    )

    filtered_data = _subset_columns(data)
    assert filtered_data.equals(
        data[["age_start", "age_end", "year_start", "year_end", "sex", "color", "value"]]
    )

    filtered_data = _subset_columns(data, color="red")
    assert filtered_data.equals(
        data[["age_start", "age_end", "year_start", "year_end", "sex", "value"]]
    )


def test_parse_artifact_path_config(
    base_config: LayeredConfigTree, test_data_dir: Path
) -> None:
    artifact_path = test_data_dir / "artifact.hdf"
    base_config.update(
        {"input_data": {"artifact_path": str(artifact_path)}}, **metadata(str(Path("/")))
    )

    assert parse_artifact_path_config(base_config) == str(artifact_path)


def test_parse_artifact_path_relative_no_source(base_config: LayeredConfigTree) -> None:
    artifact_path = "./artifact.hdf"
    base_config.update({"input_data": {"artifact_path": str(artifact_path)}})

    with pytest.raises(ValueError):
        parse_artifact_path_config(base_config)


def test_parse_artifact_path_relative(
    base_config: LayeredConfigTree, test_data_dir: Path
) -> None:
    base_config.update(
        {"input_data": {"artifact_path": "../../test_data/artifact.hdf"}},
        **metadata(__file__),
    )
    assert parse_artifact_path_config(base_config) == str(test_data_dir / "artifact.hdf")


def test_parse_artifact_path_config_fail(base_config: LayeredConfigTree) -> None:
    artifact_path = Path(__file__).parent / "not_an_artifact.hdf"
    base_config.update(
        {"input_data": {"artifact_path": str(artifact_path)}}, **metadata(str(Path("/")))
    )

    with pytest.raises(FileNotFoundError):
        parse_artifact_path_config(base_config)


def test_parse_artifact_path_config_fail_relative(base_config: LayeredConfigTree) -> None:
    base_config.update(
        {"input_data": {"artifact_path": "./not_an_artifact.hdf"}}, **metadata(__file__)
    )

    with pytest.raises(FileNotFoundError):
        parse_artifact_path_config(base_config)


def test_load_with_string_data(artifact_mock: MagicMock) -> None:
    am = ArtifactManager()
    am.artifact = artifact_mock
    am.config_filter_term = None
    assert am.load("string_data.key") == "string_data"


def test_load_with_no_data(artifact_mock: MagicMock) -> None:
    am = ArtifactManager()
    am.artifact = artifact_mock
    assert am.load("no_data.key") is None


def test_load_with_df_data(artifact_mock: MagicMock) -> None:
    am = ArtifactManager()
    am.artifact = artifact_mock
    am.config_filter_term = None
    assert isinstance(am.load("df_data.key"), pd.DataFrame)


def test_config_filter() -> None:
    df = pd.DataFrame({"year": range(1990, 2000, 1), "color": ["red", "yellow"] * 5})
    filtered = _config_filter(df, "year in [1992, 1995]")

    assert set(filtered.year) == {1992, 1995}


def test_config_filter_on_nonexistent_column() -> None:
    df = pd.DataFrame({"year": range(1990, 2000, 1), "color": ["red", "yellow"] * 5})
    filtered = _config_filter(df, "fake_col in [1992, 1995]")

    assert df.equals(filtered)


def test_validate_filter_term() -> None:
    multiple_filter_terms = "draw in [0, 1] and year > 1990"

    with pytest.raises(NotImplementedError):
        validate_filter_term(multiple_filter_terms)
