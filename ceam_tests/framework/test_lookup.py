import pytest

from datetime import datetime

import pandas as pd
import numpy as np

from ceam_tests.util import build_table, setup_simulation
from ceam.components.base_population import generate_base_population

from ceam.framework.event import Event

def test_reference_table_alignment():
    years = build_table(lambda age, sex, year: year)
    ages = build_table(lambda age, sex, year: age)
    sexes = build_table(lambda age, sex, year: sex)

    simulation = setup_simulation([generate_base_population], 10000)

    manager = simulation.tables
    years = manager.build_table(years)
    ages = manager.build_table(ages)
    sexes = manager.build_table(sexes)

    emitter = simulation.events.get_emitter('time_step__prepare')
    emitter(Event(simulation.current_time, simulation.population.population.index))

    result_years = years(simulation.population.population.index)
    result_ages = ages(simulation.population.population.index)
    result_sexes = sexes(simulation.population.population.index)

    assert np.all(result_years == simulation.current_time.year)
    assert np.all(result_ages == simulation.population.population.age)
    assert np.all(result_sexes == simulation.population.population.sex)

    simulation.current_time = datetime(simulation.current_time.year+1, simulation.current_time.month, simulation.current_time.day)
    simulation.population._population.age += 1
    emitter(Event(simulation.current_time, simulation.population.population.index))

    result_years = years(simulation.population.population.index)
    result_ages = ages(simulation.population.population.index)
    result_sexes = sexes(simulation.population.population.index)

    assert np.all(result_years == simulation.current_time.year)
    assert np.all(result_ages == simulation.population.population.age)
    assert np.all(result_sexes == simulation.population.population.sex)
