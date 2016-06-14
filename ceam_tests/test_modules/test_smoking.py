import pytest

from ceam_tests.util import simulation_factory, pump_simulation

from ceam.modules.smoking import SmokingModule
from ceam.modules.chronic_condition import ChronicConditionModule

@pytest.mark.parametrize('condition', ['ihd', 'hemorrhagic_stroke'])
def test_incidence_rate_effect(condition):
    smoking_module = SmokingModule()
    condition_module = ChronicConditionModule(condition, 'mortality_0.0.csv', 'incidence_0.7.csv', 0.01)
    simulation = simulation_factory([smoking_module, condition_module])
    pump_simulation(simulation, iterations=1)

    simulation.deregister_modules([smoking_module])

    # Base incidence rate without blood pressure
    base_incidence = simulation.incidence_rates(simulation.population, condition)

    simulation.register_modules([smoking_module])

    # Get incidence including the effect of smoking
    smoking_incidence = simulation.incidence_rates(simulation.population, condition)

    # Smoking should increase rates
    assert base_incidence.mean() < smoking_incidence.mean()
