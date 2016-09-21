# ~/ceam/ceam/modules/disease_models.py

from datetime import timedelta

from ceam import config
from ceam.framework.state_machine import Transition, State, TransitionSet
from ceam.framework.disease import DiseaseModel, DiseaseState, ExcessMortalityState, IncidenceRateTransition, ProportionTransition
# from ceam.gbd_data.gbd_ms_functions import get_angina_proportions, get_heart_failure_incidence_draws


def heart_disease_factory():
    module = DiseaseModel('ihd')

    healthy = State('healthy', key='ihd')

    # Calculate an adjusted disability weight for the acute heart attack phase that
    # accounts for the fact that our timestep is longer than the phase length
    # TODO: This doesn't account for the fact that our timestep is longer than 28 days
    timestep = config.getfloat('simulation_parameters', 'time_step')
    weight = 0.43*(2/timestep) + 0.07*(28/timestep)
    heart_attack = ExcessMortalityState('heart_attack', disability_weight=weight, dwell_time=timedelta(days=28), modelable_entity_id=1814, prevalence_meid=1814)

    #
    mild_heart_failure = ExcessMortalityState('mild_heart_failure', disability_weight=0.04, modelable_entity_id=2412, prevalence_meid=1821)
    moderate_heart_failure = ExcessMortalityState('moderate_heart_failure', disability_weight=0.07, modelable_entity_id=2412, prevalence_meid=1822)
    severe_heart_failure = ExcessMortalityState('severe_heart_failure', disability_weight=0.18, modelable_entity_id=2412, prevalence_meid=1823)

    asymptomatic_angina = ExcessMortalityState('asymptomatic_angina', disability_weight=0.0, modelable_entity_id=1817, prevalence_meid=3102)
    mild_angina = ExcessMortalityState('mild_angina', disability_weight=0.03, modelable_entity_id=1817, prevalence_meid=1818)
    moderate_angina = ExcessMortalityState('moderate_angina', disability_weight=0.08, modelable_entity_id=1817, prevalence_meid=1819)
    severe_angina = ExcessMortalityState('severe_angina', disability_weight=0.17, modelable_entity_id=1817, prevalence_meid=1820)

    # em 9/21: asymptomatic diseases have a dis weight of 0
    # @alecwd: do you know where the .08 came from?
    asymptomatic_ihd = ExcessMortalityState('asymptomatic_ihd', disability_weight=0.00, modelable_entity_id=3233, prevalence_meid=3233)

    heart_attack_transition = IncidenceRateTransition(heart_attack, 'heart_attack', modelable_entity_id=1814)
    healthy.transition_set.append(heart_attack_transition)

    heart_failure_buckets = TransitionSet(allow_null_transition=False, key="heart_failure_split")
    heart_failure_buckets.extend([
        ProportionTransition(mild_heart_failure, proportion=0.182074),
        ProportionTransition(moderate_heart_failure, proportion=0.149771),
        ProportionTransition(severe_heart_failure, proportion=0.402838),
        ])

    angina_buckets = TransitionSet(allow_null_transition=False, key="angina_split")
    angina_buckets.extend([
        ProportionTransition(asymptomatic_angina, proportion=0.304553),
        ProportionTransition(mild_angina, proportion=0.239594),
        ProportionTransition(moderate_angina, proportion=0.126273),
        ProportionTransition(severe_angina, proportion=0.32958),
        ])
    healthy.transition_set.append(IncidenceRateTransition(angina_buckets, 'non_mi_angina', modelable_entity_id=1817))

    heart_attack.transition_set.allow_null_transition=False

    # post-mi transitions
    # TODO: Figure out if we can pass in me_id here to get incidence for the correct cause of heart failure
    # TODO: Figure out how to make asymptomatic ihd be equal to whatever is left after people get heart failure and angina
    heart_attack.transition_set.append(ProportionTransition(heart_failure_buckets, get_heart_failure_incidence_draws(location_id, year_start, year_end, me_id))
    heart_attack.transition_set.append(ProportionTransition(angina_buckets, get_angina_proportions(start_year, end_year)))
    heart_attack.transition_set.append(ProportionTransition(asymptomatic_ihd, proportion=)

    mild_heart_failure.transition_set.append(heart_attack_transition)
    moderate_heart_failure.transition_set.append(heart_attack_transition)
    severe_heart_failure.transition_set.append(heart_attack_transition)
    asymptomatic_angina.transition_set.append(heart_attack_transition)
    mild_angina.transition_set.append(heart_attack_transition)
    moderate_angina.transition_set.append(heart_attack_transition)
    severe_angina.transition_set.append(heart_attack_transition)
    asymptomatic_ihd.transition_set.append(heart_attack_transition)

    module.states.extend([healthy, heart_attack, mild_heart_failure, moderate_heart_failure, severe_heart_failure, asymptomatic_angina, mild_angina, moderate_angina, severe_angina, asymptomatic_ihd])
    return module


def stroke_factory():
    module = DiseaseModel('hemorrhagic_stroke')

    healthy = State('healthy', key='hemorrhagic_stroke')
    # TODO: disability weight for stroke
    hemorrhagic_stroke = ExcessMortalityState('hemorrhagic_stroke', disability_weight=0.32, dwell_time=timedelta(days=28), modelable_entity_id=9311)
    ischemic_stroke = ExcessMortalityState('ischemic_stroke', disability_weight=0.32, dwell_time=timedelta(days=28), modelable_entity_id=9310)
    chronic_stroke = ExcessMortalityState('chronic_stroke', disability_weight=0.32, modelable_entity_id=9312)

    hemorrhagic_transition = IncidenceRateTransition(hemorrhagic_stroke, 'hemorrhagic_stroke', modelable_entity_id=9311)
    ischemic_transition = IncidenceRateTransition(ischemic_stroke, 'ischemic_stroke', modelable_entity_id=9310)
    healthy.transition_set.extend([hemorrhagic_transition, ischemic_transition])

    hemorrhagic_stroke.transition_set.append(Transition(chronic_stroke))
    ischemic_stroke.transition_set.append(Transition(chronic_stroke))

    chronic_stroke.transition_set.extend([hemorrhagic_transition, ischemic_transition])

    module.states.extend([healthy, hemorrhagic_stroke, ischemic_stroke, chronic_stroke])

    return module


# End.
