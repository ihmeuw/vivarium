import pytest

from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from ceam_tests.util import build_table, setup_simulation, generate_test_population

from ceam.framework.event import Event

def test_uniterpolated_table_alignment():
    years = build_table(lambda age, sex, year: year)
    ages = build_table(lambda age, sex, year: age)
    sexes = build_table(lambda age, sex, year: sex)

    simulation = setup_simulation([generate_test_population], 10000)

    manager = simulation.tables
    years = manager.build_table(years, key_columns=('age', 'sex', 'year'), parameter_columns=())
    ages = manager.build_table(ages, key_columns=('age', 'sex', 'year'), parameter_columns=())
    sexes = manager.build_table(sexes, key_columns=('age', 'sex', 'year'), parameter_columns=())

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

def test_interpolated_tables():
    years = build_table(lambda age, sex, year: year)
    ages = build_table(lambda age, sex, year: age)
    one_d_age = ages.copy()
    del one_d_age['year']
    one_d_age = one_d_age.drop_duplicates()

    simulation = setup_simulation([generate_test_population], 10000)
    manager = simulation.tables
    years = manager.build_table(years)
    ages = manager.build_table(ages)
    one_d_age = manager.build_table(one_d_age, parameter_columns=('age',))

    result_years = years(simulation.population.population.index)
    result_ages = ages(simulation.population.population.index)
    result_ages_1d = one_d_age(simulation.population.population.index)

    fractional_year = simulation.current_time.year
    fractional_year += simulation.current_time.timetuple().tm_yday / 365.25

    assert np.allclose(result_years, fractional_year)
    assert np.allclose(result_ages, simulation.population.population.age)
    assert np.allclose(result_ages_1d, simulation.population.population.age)

    simulation.current_time += timedelta(days=30.5 * 125)
    simulation.population._population.age += 125/12
    simulation.population._population.fractional_age += 125/12

    result_years = years(simulation.population.population.index)
    result_ages = ages(simulation.population.population.index)
    result_ages_1d = one_d_age(simulation.population.population.index)

    fractional_year = simulation.current_time.year
    fractional_year += simulation.current_time.timetuple().tm_yday / 365.25

    assert np.allclose(result_years, fractional_year)
    assert np.allclose(result_ages, simulation.population.population.age)
    assert np.allclose(result_ages_1d, simulation.population.population.age)

def test_interpolated_tables_without_uniterpolated_columns():
    years = build_table(lambda age, sex, year: year)
    del years['sex']
    years = years.drop_duplicates()

    simulation = setup_simulation([generate_test_population], 10000)
    manager = simulation.tables
    years = manager.build_table(years, key_columns=(), parameter_columns=('year', 'age',))

    result_years = years(simulation.population.population.index)

    fractional_year = simulation.current_time.year
    fractional_year += simulation.current_time.timetuple().tm_yday / 365.25

    assert np.allclose(result_years, fractional_year)

    simulation.current_time += timedelta(days=30.5 * 125)

    result_years = years(simulation.population.population.index)

    fractional_year = simulation.current_time.year
    fractional_year += simulation.current_time.timetuple().tm_yday / 365.25

    assert np.allclose(result_years, fractional_year)

def test_interpolated_tables__exact_values_at_input_points():
    years = build_table(lambda age, sex, year: year)
    input_years = years.year.unique()

    simulation = setup_simulation([generate_test_population], 10000)
    manager = simulation.tables
    years = manager.build_table(years)

    for year in input_years:
        simulation.current_time = datetime(year=year, month=1, day=1)
        assert np.allclose(years(simulation.population.population.index), simulation.current_time.year + 1/365, rtol=1.e-5)
