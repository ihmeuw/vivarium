import uuid
from functools import partial
from collections import defaultdict

import pandas as pd

from .event import listens_for
from .population import uses_columns

class TableView:
    def __init__(self, manager, key, index, columns):
        self.manager = manager
        self.index = index
        self.column_map = {key + '_' + c:c for c in columns}

    def __call__(self, pop):
        if isinstance(pop, pd.Index):
            index = pop
        else:
            index = pop.index
        result = self.manager._current_table[self.index].ix[index, list(self.column_map.keys())]

        if len(result.columns) > 1:
            return result.rename(columns=self.column_map)
        else:
            return result[result.columns[0]]

class MergedTableManager:
    def __init__(self):
        self._base_table = defaultdict(lambda: None)
        self._current_table = defaultdict(lambda: None)
        self.last_year = None

    def uses(self, table_name):
        def wrapper(consumer):
            def inner(*args, **kwargs):
                def getter(population, column):
                    return self._current_table.ix[population.simulant_id, table_name + '_' +column]
                args = list(args) + [getter]
                return consumer(*args, **kwargs)
            return inner
        return wrapper

    def build_table(self, data, index=('age', 'sex', 'year')):
        index = tuple(index)
        table = data
        key = str(uuid.uuid4())
        columns = table.columns
        column_map = {c:key + '_' + c for c in table.columns if c not in index}
        table = table.rename(columns=column_map)
        if 'sex' in table:
            table['sex'] = table.sex.astype('category')
        if self._base_table[index] is None:
            self._base_table[index] = table
        else:
            self._base_table[index] = self._base_table[index].merge(table, on=index, how='inner')

        return TableView(self, key, index, [c for c in columns if c not in index])

    @listens_for('time_step__prepare')
    @uses_columns(['age', 'sex'])
    def track_year(self, event):
        if self.last_year is None or self.last_year < event.time.year:
            for index, base_table in self._base_table.items():
                if 'year' in index:
                    base_table = base_table[base_table.year == event.time.year].drop('year', axis=1)
                    merge_index = list(set(index) - {'year'})
                else:
                    merge_index = list(index)
                current_table = base_table.merge(event.population[merge_index], on=merge_index)
                current_table['simulant_id'] = event.population.index
                current_table = current_table.set_index('simulant_id').sort_index()
                self._current_table[index] = current_table
            self.last_year = event.time.year

    def setup_components(self, components):
        pass
