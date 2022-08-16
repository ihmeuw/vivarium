from pathlib import Path

from vivarium.interface.utilities import get_output_model_name_string

_MODEL_SPEC_STEM = "model_spec_name"
_ARTIFACT_STEM = "artifact_name"
_ARTIFACT_FROM_MODEL_SPEC_STEM = f"{_ARTIFACT_STEM}_from_model_spec"

_ARTIFACT_PATH = Path(f"/totally/fake/path/for/artifact/{_ARTIFACT_STEM}.hdf")
_MODEL_SPEC_ARTIFACT_PATH = Path(f"/totally/fake/path/for/artifact/{_ARTIFACT_FROM_MODEL_SPEC_STEM}.hdf")

_MODEL_SPEC_CONTENTS_WITHOUT = """
configuration:
    input_data:
        input_draw_number: 0
"""

_MODEL_SPEC_CONTENTS_WITH = (_MODEL_SPEC_CONTENTS_WITHOUT + f"""
        artifact_path: '{_MODEL_SPEC_ARTIFACT_PATH}'
""")


def _write_file(path: Path, contents: str):
    with open(path, "w") as file:
        file.write(contents)


def test_get_output_model_name_string(tmp_path):
    # Three cases to test:
    # 1. Given an input artifact path, use that
    # 2. Without an input artifact path, but given model spec with artifact, use that
    # 3. Without the things in 1 and 2, choose the stem of the model spec

    # Write contents to tmp_path/model spec
    model_spec_path_with_artifact = Path(f"{tmp_path}/{_MODEL_SPEC_STEM}_with.yaml")
    model_spec_path_without_artifact = Path(f"{tmp_path}/{_MODEL_SPEC_STEM}.yaml")
    _write_file(model_spec_path_without_artifact, _MODEL_SPEC_CONTENTS_WITHOUT)
    _write_file(model_spec_path_with_artifact, _MODEL_SPEC_CONTENTS_WITH)

    inputs = [
        (_ARTIFACT_PATH, model_spec_path_with_artifact),
        (None, model_spec_path_with_artifact),
        (None, model_spec_path_without_artifact),
    ]

    outputs = [get_output_model_name_string(*i) for i in inputs]

    expected_outputs = [
        _ARTIFACT_STEM,
        _ARTIFACT_FROM_MODEL_SPEC_STEM,
        _MODEL_SPEC_STEM,
    ]
    assert outputs == expected_outputs
