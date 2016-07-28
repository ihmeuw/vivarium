# ~/ceam/ceam_tests/test_modules/test_smoking.py

import pytest

from ceam_tests.util import simulation_factory, pump_simulation

from ceam.modules.smoking import SmokingModule
from ceam.modules.disease_models import simple_ihd_factory, hemorrhagic_stroke_factory


@pytest.mark.parametrize('condition_module, rate_label', [(simple_ihd_factory(), 'heart_attack'), (hemorrhagic_stroke_factory(), 'hemorrhagic_stroke')])
def test_incidence_rate_effect(condition_module, rate_label):
    smoking_module = SmokingModule()
    simulation = simulation_factory([smoking_module, condition_module])
    pump_simulation(simulation, iterations=1)

    simulation.remove_children([smoking_module])

    # Base incidence rate without blood pressure
    base_incidence = simulation.incidence_rates(simulation.population, rate_label)

    simulation.add_children([smoking_module])

    # Get incidence including the effect of smoking
    smoking_incidence = simulation.incidence_rates(simulation.population, rate_label)

    # Smoking should increase rates
    assert base_incidence.mean() < smoking_incidence.mean()


# End.
