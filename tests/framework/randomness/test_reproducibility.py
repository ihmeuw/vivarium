import subprocess

import pandas as pd


def test_reproducibility(tmp_path, disease_model_spec):

    results_dir = tmp_path / "repro_check"
    results_dir.mkdir()

    cmd = f"simulate run {str(disease_model_spec)} -o {str(results_dir)}"

    subprocess.run(
        cmd,
        shell=True,
        check=True,
    )

    subprocess.run(
        cmd,
        shell=True,
        check=True,
    )

    files = [file for file in results_dir.rglob("**/*.hdf")]
    assert len(files) == 2
    df1 = pd.read_hdf(files[0]).drop(columns="simulation_run_time")
    df2 = pd.read_hdf(files[1]).drop(columns="simulation_run_time")

    assert df1.equals(df2)
