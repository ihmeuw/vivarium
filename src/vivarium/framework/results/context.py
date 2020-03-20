from typing import Callable, Generator, List, Union

import pandas as pd

from vivarium.framework.results import (FormattingStrategy, BinningStrategy, MappingStrategy,
                                        MappingStrategyPool, ResultsProducerStrategyPool)


class ResultsContext:

    def __init__(self, formatting_strategy: FormattingStrategy):
        self._format = formatting_strategy
        self._mapper = MappingStrategyPool()
        self._producer = ResultsProducerStrategyPool()

    def add_binning_strategy(self, target: str, result_column: str, bins: List, labels: List[str], **cut_kwargs):
        self._mapper.add_strategy(BinningStrategy(target, result_column, bins, labels, **cut_kwargs))

    def add_mapping_strategy(self, target: Union[str, List[str]], result_column: str,
                             mapper: Callable, is_vectorized: bool):
        self._mapper.add_strategy(MappingStrategy(target, result_column, mapper, is_vectorized))

    def get_results(self, data: pd.DataFrame, event_name: str) -> Generator:
        data = self._mapper.expand_data(data)
        for result in self._producer.produce_results(event_name, data):
            yield self._format(result)


