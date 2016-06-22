import pytest

import pandas as pd

from ceam.modules.opportunistic_screening import _hypertensive_categories, OpportunisticScreeningModule

def _population_factory():
    population = []
    population.append((40, 130)) # Normotensive, below 60
    population.append((60, 145)) # Normotensive, exactly 60
    population.append((70, 145)) # Normotensive, above 60

    population.append((40, 140)) # Hypertensive, below 60
    population.append((40, 145)) # Hypertensive, below 60
    population.append((60, 170)) # Hypertensive, exatly 60
    population.append((70, 150)) # Hypertensive, above 60
    population.append((70, 155)) # Hypertensive, above 60

    population.append((40, 185)) # Severe hypertensive, below 60
    population.append((70, 185)) # Severe hypertensive, above 60

    return pd.DataFrame(population, columns=['age', 'systolic_blood_pressure'])


def test_hypertensive_categories():
    population = _population_factory()

    normotensive, hypertensive, severe_hypertension = _hypertensive_categories(population)

    assert len(normotensive) == 3
    assert len(hypertensive) == 5
    assert len(severe_hypertension) == 2
