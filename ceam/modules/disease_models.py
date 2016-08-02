# ~/ceam/ceam/modules/disease_models.py

from datetime import timedelta

from ceam.state_machine import Transition, State
from ceam.modules.disease import DiseaseModule, DiseaseState, ExcessMortalityState, IncidenceRateTransition


def heart_disease_factory():
    module = DiseaseModule('ihd')

    healthy = State('healthy')

    # TODO: This assumes a 30.5 day timestep which isn't guarenteed
    # TODO: This doesn't account for the fact that our timestep is longer than 28 days
    weight = 0.43*(2/30.5) + 0.07*(28/30.5)
    heart_attack = ExcessMortalityState('heart_attack', disability_weight=weight, dwell_time=timedelta(days=28), modelable_entity_id=1814)

    mild_heart_failure = ExcessMortalityState('mild_heart_failure', disability_weight=0.04, modelable_entity_id=2412)
    moderate_heart_failure = ExcessMortalityState('moderate_heart_failure', disability_weight=0.07, modelable_entity_id=2412)
    severe_heart_failure = ExcessMortalityState('severe_heart_failure', disability_weight=0.18, modelable_entity_id=2412)

    angina = ExcessMortalityState('angina', disability_weight=0.08, modelable_entity_id=1817)

    heart_attack_transition = IncidenceRateTransition(heart_attack, 'heart_attack', modelable_entity_id=1814)
    healthy.transition_set.append(heart_attack_transition)
    healthy.transition_set.append(IncidenceRateTransition(angina, 'non_mi_angina', modelable_entity_id=1817))

    heart_attack.transition_set.allow_null_transition=False
    heart_attack.transition_set.append(Transition(mild_heart_failure))
    heart_attack.transition_set.append(Transition(moderate_heart_failure))
    heart_attack.transition_set.append(Transition(severe_heart_failure))
    heart_attack.transition_set.append(Transition(angina))

    mild_heart_failure.transition_set.append(heart_attack_transition)
    moderate_heart_failure.transition_set.append(heart_attack_transition)
    severe_heart_failure.transition_set.append(heart_attack_transition)
    angina.transition_set.append(heart_attack_transition)

    module.states.extend([healthy, heart_attack, mild_heart_failure, moderate_heart_failure, severe_heart_failure, angina])
    return module


def simple_ihd_factory():
    module = DiseaseModule('ihd')

    healthy = State('healthy')
    # TODO: disability weight for heart attack
    heart_attack = ExcessMortalityState('heart_attack', disability_weight=0.439, dwell_time=timedelta(days=28), modelable_entity_id=1814)
    chronic_ihd = ExcessMortalityState('chronic_ihd', disability_weight=0.08, modelable_entity_id=2412)

    heart_attack_transition = IncidenceRateTransition(heart_attack, 'heart_attack', modelable_entity_id=1814)
    healthy.transition_set.append(heart_attack_transition)

    heart_attack.transition_set.append(Transition(chronic_ihd))

    chronic_ihd.transition_set.append(heart_attack_transition)

    module.states.extend([healthy, heart_attack, chronic_ihd])

    return module


def stroke_factory():
    module = DiseaseModule('hemorrhagic_stroke')

    healthy = State('healthy')
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
