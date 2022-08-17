from pathlib import Path

import pytest

from vivarium.interface.utilities import get_output_model_name_string

_MODEL_SPEC_STEM = "model_spec_name"
_ARTIFACT_STEM = "artifact_name"
_ARTIFACT_FROM_MODEL_SPEC_STEM = f"{_ARTIFACT_STEM}_from_model_spec"

_ARTIFACT_PATH = Path(f"/totally/fake/path/for/artifact/{_ARTIFACT_STEM}.hdf")
_MODEL_SPEC_ARTIFACT_PATH = Path(
    f"/totally/fake/path/for/artifact/{_ARTIFACT_FROM_MODEL_SPEC_STEM}.hdf"
)

_MODEL_SPEC_CONTENTS_WITHOUT = """
configuration:
    input_data:
        input_draw_number: 0
"""

_MODEL_SPEC_CONTENTS_WITH = (
    _MODEL_SPEC_CONTENTS_WITHOUT
    + f"""
        artifact_path: '{_MODEL_SPEC_ARTIFACT_PATH}'
"""
)


def _write_file(path: Path, contents: str):
    with open(path, "w") as file:
        file.write(contents)


@pytest.mark.parametrize(
    "artifact_path, model_spec_filename, contents, expected_output",
    [
        (
            # Given an input artifact path, use that
            _ARTIFACT_PATH,
            f"{_MODEL_SPEC_STEM}_with.yaml",
            _MODEL_SPEC_CONTENTS_WITH,
            _ARTIFACT_STEM,
        ),
        (
            # Without an input artifact path, but given model spec with artifact, use that
            None,
            f"{_MODEL_SPEC_STEM}_with.yaml",
            _MODEL_SPEC_CONTENTS_WITH,
            _ARTIFACT_FROM_MODEL_SPEC_STEM,
        ),
        (
            # Without the things in previous parameters, choose the stem of the model spec
            None,
            f"{_MODEL_SPEC_STEM}.yaml",
            _MODEL_SPEC_CONTENTS_WITHOUT,
            _MODEL_SPEC_STEM,
        ),
    ],
)
def test_get_output_model_name_string(
    artifact_path,
    model_spec_filename,
    contents,
    expected_output,
    tmp_path,
):
    model_spec_path = Path(f"{tmp_path}/{model_spec_filename}")
    _write_file(model_spec_path, contents)

    output = get_output_model_name_string(artifact_path, model_spec_path)

    assert output == expected_output
