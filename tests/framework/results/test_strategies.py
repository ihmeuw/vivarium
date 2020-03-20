import pytest

from vivarium.framework.results import FormattingStrategy


def test_formatting_strategy_initialization():
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        FormattingStrategy('measure')
