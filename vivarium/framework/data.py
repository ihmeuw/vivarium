from datetime import datetime

import pandas as pd

from vivarium.framework.engine import _get_time

import logging
_log = logging.getLogger(__name__)

class Artifact:
    def setup(self, builder):
        self._loading_start_time = datetime.now()
        self.end_time = _get_time("end", builder.configuration.simulation_parameters)
        self.start_time = _get_time("start", builder.configuration.simulation_parameters)
        self.draw = builder.configuration.run_configuration.input_draw_number
        self.location = builder.configuration.input_data.location_id

        self._hdf = pd.HDFStore(builder.configuration.input_data.artifact_path)
        builder.event.register_listener('post_setup', self.close)
        self._cache = {}

    def load(self, entity_path, keep_age_group_edges=False, **column_filters):
        _log.debug(f"loading {entity_path}")
        cache_key = (entity_path, tuple(sorted(column_filters.items())))
        if cache_key in self._cache:
            _log.debug("    from cache")
            return self._cache[cache_key]
        else:
            group = '/'+entity_path.replace('.','/')
            #TODO: Is there a better way to get the columns without loading  much data?
            columns = self._hdf.select(group, stop=1).columns
            terms = []
            default_filters = {
                'draw': self.draw,
                'year': [f">= {self.start_time.year}", f"<= {self.end_time.year}"],
            }
            default_filters.update(column_filters)
            column_filters = default_filters
            for column, condition in column_filters.items():
                if column in columns:
                    if not isinstance(condition, (list, tuple)):
                        condition = [condition]
                    for c in condition:
                        if isinstance(c, str):
                            terms.append(f"{column} {c}")
                        else:
                            terms.append(f"{column} = {c}")
            columns_to_remove = set(column_filters.keys())
            if "location_id" not in column_filters and "location_id" in columns:
                #TODO I think this is a sign I should be handling queries differently
                terms.append(f"location_id == {self.location} | location_id == 1")
                columns_to_remove.add("location_id")
            data = pd.read_hdf(self._hdf, group, where=terms if terms else None)
            # FIXME I don't like how special year is
            columns_to_remove = columns_to_remove - {"year"}
            # FIXME same with the age group columns
            if not keep_age_group_edges:
                columns_to_remove = columns_to_remove | {"age_group_start", "age_group_end"}
            columns_to_remove = columns_to_remove.intersection(columns)

            data = data.drop(columns=columns_to_remove)
            self._cache[(entity_path, None)] = data
        self._cache[cache_key] = data
        return data

    def close(self, event):
        self._hdf.close()
        self._cache = {}
        _log.debug(f"Data loading took at most {datetime.now() - self._loading_start_time} seconds")
