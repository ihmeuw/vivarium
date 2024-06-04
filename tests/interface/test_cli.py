from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from tests import HARRY_POTTER_CONFIG
from vivarium.interface.cli import simulate


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def model_spec(base_config, tmp_path) -> str:
    base_config.update(HARRY_POTTER_CONFIG)
    model_spec = {}
    model_spec["configuration"] = base_config.to_dict()
    model_spec["components"] = {
        "tests": [
            "Hogwarts()",
            "HousePointsObserver()",
            "NoStratificationsQuidditchWinsObserver()",
            "QuidditchWinsObserver()",
            "HogwartsResultsStratifier()",
        ],
    }
    filepath = f"{tmp_path}/model_spec.yaml"
    with open(filepath, "w") as f:
        yaml.dump(model_spec, f)
    return filepath


@pytest.mark.parametrize("cli_args", [(), ("-i",), ("-l",), ("-i", "-l")])
def test_simulate_run(runner, model_spec, hdf_file_path, cli_args):
    output_dir = "/".join(model_spec.split("/")[:-1])
    extra_args = []
    if "-i" in cli_args:
        extra_args.extend(["-i", hdf_file_path])
    if "-l" in cli_args:
        extra_args.extend(["-l", "Hogwarts"])
    args = ["run", model_spec, "-o", output_dir] + extra_args
    sim = runner.invoke(simulate, args)
    assert sim.exit_code == 0
    results_dir = list(Path(output_dir).rglob("*/simulation.log"))[0].parent
    assert {file.name for file in results_dir.rglob("*.parquet")} == {
        "house_points.parquet",
        "quidditch_wins.parquet",
        "no_stratifications_quidditch_wins.parquet",
    }
    with open(results_dir / "metadata.yaml", "r") as f:
        metadata = yaml.safe_load(f)

    assert set(metadata) == set(
        ["input_draw", "random_seed", "simulation_run_time", "artifact_path", "location"]
    )
    assert metadata["simulation_run_time"] > 0.0
    if "-i" in cli_args:
        assert metadata["artifact_path"] == hdf_file_path
    else:
        assert metadata["artifact_path"] is None
    if "-l" in cli_args:
        assert metadata["location"] == "Hogwarts"
    else:
        assert metadata["location"] is None
