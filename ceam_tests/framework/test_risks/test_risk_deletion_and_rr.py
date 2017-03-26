# ~/diarrhea_mvs/ceam/ceam_tests/framework/test_risks/test_risk_deletion_and_rr.py

import pandas as pd
import numpy as np
import pytest
from datetime import datetime
from ceam_tests.util import build_table, setup_simulation, generate_test_population, pump_simulation
from ceam_public_health.components.diarrhea_disease_model import diarrhea_factory
from ceam_public_health.components.risks.categorical_risk_handler import CategoricalRiskHandler


##########################################################
# Step 1. Create fixtures for risk deletion and rr tests #
##########################################################


# FIXME: @alecwd: Is 'object' the correct way to classify 'simulation'?
@pytest.fixture
def set_up_test_parameters(simulation, multiple_risks_test):
    """
    Fixture that sets up a base incidence, PAF, and RR.

    Parameters
    ----------
    simulation: object @alecwd: Is 'object' the correct way to classify 'simulation'?
        CEAM simulation object
    
    multiple_risks_test: Bool
        Set to true if doing a test of a simulation with multiple risk factors 
    """
    # Set base incidence of rotavirus to 10 arbitrarily
    simulation_incidence = simulation.values.get_rate('base_incidence_rate.diarrhea_due_to_rotaviral_entiritis')
    incidence_rota = build_table(10, ['age', 'year', 'sex', 'rate'])
    simulation_incidence.source = simulation.tables.build_table(incidence_rota)

    # Set PAF arbitrarily to .75 for stunting
    stunting_paf = build_table(.75, ['age', 'year', 'sex', 'PAF'])
    simulation_stunting_paf = simulation.values.get_value('paf_of_stunting_on_diarrhea_due_to_rotaviral_entiritis')
    simulation_stunting_paf.source = simulation.tables.build_table(stunting_paf)

    # Set RR arbitrarily to 2 for severe stunting, 1 for all other categories
    stunting_rr = build_table(2, ['age', 'year', 'sex', 'cat1'])
    stunting_rr['cat2'] = 1
    stunting_rr['cat3'] = 1
    stunting_rr['cat4'] = 1

    simulation_stunting_relative_risk = simulation.values.get_value('relative_risk_of_stunting_on_diarrhea_due_to_rotaviral_entiritis')    
    simulation_stunting_relative_risk.source = simulation.tables.build_table(stunting_rr)

    # If testing a simulation with 2 risk factors, set PAF and RR for unsafe_sanitation as well
    if multiple_risks_test:
        # Set PAF arbitrarily to .25 for unsafe sanitation
        unsafe_sanitation_paf = build_table(.25, ['age', 'year', 'sex', 'PAF'])
        simulation_unsafe_sanitation_paf = simulation.values.get_value('paf_of_unsafe_sanitation_on_diarrhea_due_to_rotaviral_entiritis')
        simulation_unsafe_sanitation_paf.source = simulation.tables.build_table(unsafe_sanitation_paf)

        # Set RR arbitrarily to 2 for severe unsafe_sanitation, 1 for all other categories
        unsafe_sanitation_rr = build_table(2, ['age', 'year', 'sex', 'cat1'])
        unsafe_sanitation_rr['cat2'] = 1
        unsafe_sanitation_rr['cat3'] = 1

        simulation_unsafe_sanitation_relative_risk = simulation.values.get_value('relative_risk_of_unsafe_sanitation_on_diarrhea_due_to_rotaviral_entiritis')
        simulation_unsafe_sanitation_relative_risk.source = simulation.tables.build_table(unsafe_sanitation_rr)

    return simulation


# FIXME: @alecwd: Is 'object' the correct way to classify 'simulation'?
@pytest.fixture
def set_up_exposures(simulation, multiple_risks_test, full_exposure):
    """
    Parameters
    ----------
    simulation: object @alecwd: Is 'object' the correct way to classify 'simulation'?
        CEAM simulation object

    multiple_risks_test: Bool
        Set to true if doing a test of a simulation with multiple risk factors

    full_exposure: Bool
        set to true to set exposure to the most severe category to 100%, false to set exposure to the least severe category to 100%
    """
    if not full_exposure:
        stunting_exposure = build_table(0, ['age', 'year', 'sex', 'cat1'])
        stunting_exposure['cat2'] = 0
        stunting_exposure['cat3'] = 0
        stunting_exposure['cat4'] = 1

        simulation_stunting_exposure = simulation.values.get_rate('stunting.exposure')
        simulation_stunting_exposure.source = simulation.tables.build_table(stunting_exposure)

    else:
        stunting_exposure = build_table(1, ['age', 'year', 'sex', 'cat1'])
        stunting_exposure['cat2'] = 0
        stunting_exposure['cat3'] = 0
        stunting_exposure['cat4'] = 0

        simulation_stunting_exposure = simulation.values.get_rate('stunting.exposure')
        simulation_stunting_exposure.source = simulation.tables.build_table(stunting_exposure)

    if multiple_risks_test:
        if not full_exposure:
            unsafe_sanitation_exposure = build_table(0, ['age', 'year', 'sex', 'cat1'])
            unsafe_sanitation_exposure['cat2'] = 0
            unsafe_sanitation_exposure['cat3'] = 1

            simulation_unsafe_sanitation_exposure = simulation.values.get_rate('unsafe_sanitation.exposure')
            simulation_unsafe_sanitation_exposure.source = simulation.tables.build_table(unsafe_sanitation_exposure)

        else:
            unsafe_sanitation_exposure = build_table(1, ['age', 'year', 'sex', 'cat1'])
            unsafe_sanitation_exposure['cat2'] = 0
            unsafe_sanitation_exposure['cat3'] = 0

            simulation_unsafe_sanitation_exposure = simulation.values.get_rate('unsafe_sanitation.exposure')
            simulation_unsafe_sanitation_exposure.source = simulation.tables.build_table(unsafe_sanitation_exposure)
   
    return simulation


#######################################################################
# Step 2. Conduct tests in a simulation that includes one risk factor #
#######################################################################

# FIXME: Expect one risk simulation tests to fail until the PAF pipeline is set up to return base incidence * (1 - PAF) for a one risk simulation


@pytest.mark.xfail
def test_risk_deletion_with_one_risk_simulation():
    simulation = setup_simulation(components=[generate_test_population, CategoricalRiskHandler(241, 'stunting')] 
                                          + diarrhea_factory(), start=datetime(2005, 1, 1))

    simulation = set_up_test_parameters(simulation, multiple_risks_test=False)

    simulation = set_up_exposures(simulation, multiple_risks_test=False, full_exposure=False)

    rota_effective_inc = simulation.values.get_rate('incidence_rate.diarrhea_due_to_rotaviral_entiritis')
    rota_inc_unexposed = rota_effective_inc(simulation.population.population.index)

    assert np.all(rota_inc_unexposed == 2.5), "risk deleted incidence rate should be equal to base rate * (1-PAF)." + \
                                              "In the case of this test the answer should be 2.5. (10 * (1-.75))"


@pytest.mark.xfail
def test_that_rrs_applied_correctly_with_one_risk_simulation():
    simulation = setup_simulation(components=[generate_test_population, CategoricalRiskHandler(241, 'stunting')]
                                          + diarrhea_factory(), start=datetime(2005, 1, 1))

    simulation = set_up_test_parameters(simulation, multiple_risks_test=False)

    simulation = set_up_exposures(simulation, multiple_risks_test=False, full_exposure=True)

    rota_effective_inc = simulation.values.get_rate('incidence_rate.diarrhea_due_to_rotaviral_entiritis')
    rota_inc_exposed = rota_effective_inc(simulation.population.population.index)

    assert np.all(rota_inc_exposed == 5), "incidence rate should be equal to base rate * (1-PAF) * RR." + \
                                            "In the case of this test the answer should be 5. (10 * (1-.75) * 2)"


#############################################################################
# Step 3. Conduct tests in a simulation that includes multiple risk factors #
#############################################################################


def test_risk_deletion_with_multiple_risk_simulation():
    simulation = setup_simulation(components=[generate_test_population, CategoricalRiskHandler(241, 'stunting'), CategoricalRiskHandler(84, 'unsafe_sanitation')] + diarrhea_factory(), start=datetime(2005, 1, 1))

    simulation = set_up_test_parameters(simulation, multiple_risks_test=True)

    simulation = set_up_exposures(simulation, multiple_risks_test=True, full_exposure=False)

    rota_effective_inc = simulation.values.get_rate('incidence_rate.diarrhea_due_to_rotaviral_entiritis')
    rota_inc_unexposed = rota_effective_inc(simulation.population.population.index)

    assert np.all(rota_inc_unexposed == 1.875), "risk deleted incidence rate should be equal to base rate * (1 - (1 - PAF1) * (1 - PAF2))." + "In the case of this test the answer should be 1.875. (10 * (1 - Joint PAF). Joint PAF = (1 - (1 - .75) * (1 - .25))"

def test_that_rrs_applied_correctly_with_multiple_risk_simulation():
    simulation = setup_simulation(components=[generate_test_population, CategoricalRiskHandler(241, 'stunting'), CategoricalRiskHandler(84, 'unsafe_sanitation')] + diarrhea_factory(), start=datetime(2005, 1, 1))

    simulation = set_up_test_parameters(simulation, multiple_risks_test=True)

    simulation = set_up_exposures(simulation, multiple_risks_test=True, full_exposure=True)

    rota_effective_inc = simulation.values.get_rate('incidence_rate.diarrhea_due_to_rotaviral_entiritis')
    rota_inc_exposed = rota_effective_inc(simulation.population.population.index)

    assert np.all(rota_inc_exposed == 7.5), "risk deleted incidence rate should be equal to base rate * (1 - (1 - PAF1) * (1 - PAF2))." + "In the case of this test the answer should be 7.5. (10 * [1 - Joint PAF] * 2 * 2). Joint PAF = (1 - [1 - .75] * [1 - .25])"

# End.
