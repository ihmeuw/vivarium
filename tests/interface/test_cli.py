from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from layered_config_tree import LayeredConfigTree

from tests.framework.results.helpers import HARRY_POTTER_CONFIG
from vivarium.interface.cli import simulate


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def model_spec(base_config: LayeredConfigTree, tmp_path: Path) -> str:
    base_config.update(HARRY_POTTER_CONFIG)
    model_spec = {}
    model_spec["configuration"] = base_config.to_dict()
    model_spec["components"] = {
        "tests.framework.results.helpers": [
            "Hogwarts()",
            "HousePointsObserver()",
            "NoStratificationsQuidditchWinsObserver()",
            "QuidditchWinsObserver()",
            "HogwartsResultsStratifier()",
        ],
    }
    (tmp_path / "model_spec").mkdir()
    filepath = tmp_path / "model_spec" / "model_spec.yaml"
    with open(filepath, "w") as f:
        yaml.dump(model_spec, f)
    return str(filepath)


def test_simulate_run(runner: CliRunner, model_spec: str, hdf_file_path: Path) -> None:
    run_parameters = {param.name for param in simulate.commands["run"].params}
    expected_parameters = {
        "model_specification",
        "artifact_path",
        "results_directory",
        "verbose",
        "quiet",
        "with_debugger",
    }
    different_params = run_parameters.symmetric_difference(expected_parameters)
    if run_parameters.symmetric_difference(expected_parameters):
        raise ValueError(
            f"Missing or unexpected parameters in simulate run: {different_params}"
        )
    output_dir = "/".join(model_spec.split("/")[:-2])
    args = ["run", model_spec, "-o", output_dir, "-i", str(hdf_file_path)]
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
        ["input_draw", "random_seed", "simulation_run_time", "artifact_path"]
    )
    assert metadata["simulation_run_time"] > 0.0
    # Ensure '-i' worked
    with open(f"{output_dir}/model_specification.yaml") as f:
        ms = yaml.safe_load(f)
    assert ms["configuration"]["input_data"]["artifact_path"] == str(hdf_file_path)
