from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
import pytest_mock
import tables
import yaml
from _pytest.logging import LogCaptureFixture
from layered_config_tree import LayeredConfigTree
from loguru import logger
from vivarium_testing_utils import FuzzyChecker

from vivarium.framework.configuration import (
    build_model_specification,
    build_simulation_configuration,
)
from vivarium.testing_utilities import metadata


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def fuzzy_checker() -> FuzzyChecker:
    return FuzzyChecker()


@pytest.fixture
def caplog(caplog: LogCaptureFixture) -> Generator[LogCaptureFixture, None, None]:
    handler_id = logger.add(caplog.handler, format="{message}")
    yield caplog
    logger.remove(handler_id)


@pytest.fixture
def base_config() -> LayeredConfigTree:
    config = build_simulation_configuration()
    config.update(
        {
            "time": {
                "start": {
                    "year": 1990,
                },
                "end": {"year": 2010},
                "step_size": 30.5,
            },
            "randomness": {"key_columns": ["entrance_time", "age"]},
        },
        **metadata(__file__, layer="model_override"),
    )
    return config


@pytest.fixture
def test_data_dir() -> Path:
    data_dir = Path(__file__).resolve().parent / "test_data"
    assert data_dir.exists(), "Test directory structure is broken"
    return data_dir


@pytest.fixture(params=[".yaml", ".yml"])
def test_spec(request: pytest.FixtureRequest, test_data_dir: Path) -> Path:
    return test_data_dir / f"mock_model_specification{request.param}"


@pytest.fixture(params=[".yaml", ".yml"])
def test_user_config(request: pytest.FixtureRequest, test_data_dir: Path) -> Path:
    return test_data_dir / f"mock_user_config{request.param}"


@pytest.fixture
def model_specification(
    mocker: pytest_mock.MockFixture, test_spec: Path, test_user_config: Path
) -> LayeredConfigTree:
    expand_user_mock = mocker.patch("vivarium.framework.configuration.Path.expanduser")
    expand_user_mock.return_value = test_user_config
    return build_model_specification(test_spec)


@pytest.fixture
def disease_model_spec(tmp_path: Path) -> Path:
    model_spec_path = (
        Path(__file__).resolve().parent.parent
        / "src/vivarium/examples/disease_model/disease_model.yaml"
    )
    with open(model_spec_path, "r") as file:
        ms = yaml.safe_load(file)

    # modify the time so as not to take so long for a unit test
    ms["configuration"]["time"]["end"]["year"] = ms["configuration"]["time"]["start"]["year"]
    ms["configuration"]["time"]["end"]["month"] = ms["configuration"]["time"]["start"][
        "month"
    ]
    ms["configuration"]["time"]["start"]["day"] = 1
    ms["configuration"]["time"]["end"]["day"] = 5
    ms["configuration"]["time"]["step_size"] = 0.5
    model_spec = tmp_path / "disease_model.yaml"

    with open(model_spec, "w") as file:
        yaml.dump(ms, file)

    return model_spec


@pytest.fixture
def hdf_file_path(tmp_path: Path, test_data_dir: Path) -> Path:
    """This file contains the following:
    Object Tree:
        / (RootGroup) ''
        /cause (Group) ''
        /population (Group) ''
        /population/age_bins (Group) ''
        /population/age_bins/table (Table(23,), shuffle, zlib(9)) ''
        /population/structure (Group) ''
        /population/structure/table (Table(1863,), shuffle, zlib(9)) ''
        /population/theoretical_minimum_risk_life_expectancy (Group) ''
        /population/theoretical_minimum_risk_life_expectancy/table (Table(10502,), shuffle, zlib(9)) ''
        /population/structure/meta (Group) ''
        /population/structure/meta/values_block_1 (Group) ''
        /population/structure/meta/values_block_1/meta (Group) ''
        /population/structure/meta/values_block_1/meta/table (Table(3,), shuffle, zlib(9)) ''
        /cause/all_causes (Group) ''
        /cause/all_causes/restrictions (EArray(166,)) ''
    """
    # Make temporary copy of file for test.
    p = tmp_path / "artifact.hdf"
    with tables.open_file(str(test_data_dir / "artifact.hdf")) as file:
        file.copy_file(str(p), overwrite=True)
    return p


@pytest.fixture
def hdf_file(hdf_file_path: Path) -> Generator[tables.file.File, None, None]:
    with tables.open_file(str(hdf_file_path)) as file:
        yield file
