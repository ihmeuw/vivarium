from pathlib import Path

import pandas as pd


def dataframe_to_csv(
    results_dir: Path, measure: str, results: pd.DataFrame, random_seed: str, input_draw: str
) -> None:
    # Add extra cols
    results[["measure"]] = measure
    results["random_seed"] = random_seed
    results["input_draw"] = input_draw
    # Sort the columns such that the stratifications (index) are first
    # and "value" is last and sort the rows by the stratifications.
    other_cols = [c for c in results.columns if c != "value"]
    results = results[other_cols + ["value"]].sort_index().reset_index()
    results.to_csv(results_dir / f"{measure}.csv", index=False)
