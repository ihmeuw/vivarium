import subprocess
from pathlib import Path

import pandas as pd


def test_reproducibility(tmp_path: Path, disease_model_spec: Path) -> None:

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

    files = [file for file in results_dir.rglob("**/*.parquet")]
    assert len(files) == 4
    for filename in ["dead", "ylls"]:
        df_paths = [file for file in files if file.stem == filename]
        df1 = pd.read_parquet(df_paths[0])
        df2 = pd.read_parquet(df_paths[1])

        assert df1.equals(df2)
