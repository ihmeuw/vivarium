import abc
from typing import Dict


class FormattingStrategy(abc.ABC):
    """Base interface for results formatting strategies."""

    def __init__(self, measure: str, **additional_keys: Dict[str, str]):
        """
        Parameters
        ----------
        measure
            The measure this strategy is formatted to produce.
        additional_keys
            Additional labels to attach to the formatted data.

        """
        self._measure = measure
        self._additional_keys = additional_keys

    @abc.abstractmethod
    def __call__(self, data: pd.DataFrame):
        pass

