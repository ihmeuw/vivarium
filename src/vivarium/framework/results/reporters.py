from pathlib import Path

import pandas as pd


def dataframe_to_csv(
    results_dir: Path, measure: str, df: pd.DataFrame, random_seed: str, input_draw: str
) -> None:
    # Add extra cols
    df[["measure"]] = measure
    df["random_seed"] = random_seed
    df["input_draw"] = input_draw
    # Sort the columns such that the stratifications (index) are first
    # and "value" is last and sort the rows by the stratifications.
    other_cols = [c for c in df.columns if c != "value"]
    df = df[other_cols + ["value"]].sort_index().reset_index()
    df.to_csv(results_dir / f"{measure}.csv", index=False)
