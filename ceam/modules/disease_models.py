# ~/ceam/ceam/modules/disease_models.py

from datetime import timedelta

from ceam.state_machine import Transition, State
from ceam.modules.disease import DiseaseModule, DiseaseState, ExcessMortalityState, IncidenceRateTransition


def heart_disease_factory():
    module = DiseaseModule('ihd')

    healthy = State('healthy')
    # TODO: disability weight for heart attack
    heart_attack = ExcessMortalityState('heart_attack', disability_weight=0.439, dwell_time=timedelta(days=28), excess_mortality_table='mi_acute_excess_mortality.csv')

    mild_heart_failure = ExcessMortalityState('mild_heart_failure', disability_weight=0.08, excess_mortality_table='ihd_mortality_rate.csv')
    moderate_heart_failure = ExcessMortalityState('moderate_heart_failure', disability_weight=0.08, excess_mortality_table='ihd_mortality_rate.csv')
    severe_heart_failure = ExcessMortalityState('severe_heart_failure', disability_weight=0.08, excess_mortality_table='ihd_mortality_rate.csv')
    angina = ExcessMortalityState('angina', disability_weight=0.08, excess_mortality_table='ihd_mortality_rate.csv')

    heart_attack_transition = IncidenceRateTransition(heart_attack, 'heart_attack', 'ihd_incidence_rates.csv')
    angina_transition = IncidenceRateTransition(angina, 'angina', 'ihd_incidence_rates.csv')
    healthy.transition_set.add(heart_attack_transition)
    healthy.transition_set.add(angina_transition)

    heart_attack.transition_set.allow_null_transition=False
    heart_attack.transition_set.add(Transition(mild_heart_failure))
    heart_attack.transition_set.add(Transition(moderate_heart_failure))
    heart_attack.transition_set.add(Transition(severe_heart_failure))
    heart_attack.transition_set.add(Transition(angina))

    mild_heart_failure.transition_set.add(heart_attack_transition)
    moderate_heart_failure.transition_set.add(heart_attack_transition)
    severe_heart_failure.transition_set.add(heart_attack_transition)
    angina.transition_set.add(heart_attack_transition)

    module.states.update([healthy, heart_attack, mild_heart_failure, moderate_heart_failure, severe_heart_failure, angina])
    return module


def simple_ihd_factory():
    module = DiseaseModule('ihd')

    healthy = State('healthy')
    # TODO: disability weight for heart attack
    heart_attack = ExcessMortalityState('heart_attack', disability_weight=0.439, dwell_time=timedelta(days=28), excess_mortality_table='mi_acute_excess_mortality.csv')
    chronic_ihd = ExcessMortalityState('chronic_ihd', disability_weight=0.08, excess_mortality_table='ihd_mortality_rate.csv')

    heart_attack_transition = IncidenceRateTransition(heart_attack, 'heart_attack', 'ihd_incidence_rates.csv')
    healthy.transition_set.add(heart_attack_transition)

    heart_attack.transition_set.add(Transition(chronic_ihd))

    chronic_ihd.transition_set.add(heart_attack_transition)

    module.states.update([healthy, heart_attack, chronic_ihd])

    return module


def hemorrhagic_stroke_factory():
    module = DiseaseModule('hemorrhagic_stroke')

    healthy = State('healthy')
    # TODO: disability weight for stroke
    stroke = ExcessMortalityState('hemorrhagic_stroke', disability_weight=0.92, dwell_time=timedelta(days=28), excess_mortality_table='acute_hem_stroke_excess_mortality.csv')
    chronic_stroke = ExcessMortalityState('chronic_stroke', disability_weight=0.31, excess_mortality_table='chronic_hem_stroke_excess_mortality.csv')

    stroke_transition = IncidenceRateTransition(stroke, 'hemorrhagic_stroke', 'hem_stroke_incidence_rates.csv')
    healthy.transition_set.add(stroke_transition)

    stroke.transition_set.add(Transition(chronic_stroke))

    chronic_stroke.transition_set.add(stroke_transition)

    module.states.update([healthy, stroke, chronic_stroke])

    return module


# End.
