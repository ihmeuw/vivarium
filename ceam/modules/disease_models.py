# ~/ceam/ceam/modules/disease_models.py

from datetime import timedelta

from ceam.state_machine import Transition, State, TransitionSet
from ceam.modules.disease import DiseaseModule, DiseaseState, ExcessMortalityState, IncidenceRateTransition, ProportionTransition


def heart_disease_factory():
    module = DiseaseModule('ihd')

    healthy = State('healthy')

    # Calculate an adjusted disability weight for the acute heart attack phase that
    # accounts for the fact that our timestep is longer than the phase length
    # TODO: This assumes a 30.5 day timestep which isn't guaranteed
    # TODO: This doesn't account for the fact that our timestep is longer than 28 days
    weight = 0.43*(2/30.5) + 0.07*(28/30.5)
    heart_attack = ExcessMortalityState('heart_attack', disability_weight=weight, dwell_time=timedelta(days=28), modelable_entity_id=1814, prevalence_meid=1814)

    mild_heart_failure = ExcessMortalityState('mild_heart_failure', disability_weight=0.04, modelable_entity_id=2412, prevalence_meid=1821)
    moderate_heart_failure = ExcessMortalityState('moderate_heart_failure', disability_weight=0.07, modelable_entity_id=2412, prevalence_meid=1822)
    severe_heart_failure = ExcessMortalityState('severe_heart_failure', disability_weight=0.18, modelable_entity_id=2412, prevalence_meid=1823)

    asymptomatic_angina = ExcessMortalityState('asymptomatic_angina', disability_weight=0.0, modelable_entity_id=1817, prevalence_meid=3102)
    mild_angina = ExcessMortalityState('mild_angina', disability_weight=0.03, modelable_entity_id=1817, prevalence_meid=1818)
    moderate_angina = ExcessMortalityState('moderate_angina', disability_weight=0.08, modelable_entity_id=1817, prevalence_meid=1819)
    severe_angina = ExcessMortalityState('severe_angina', disability_weight=0.17, modelable_entity_id=1817, prevalence_meid=1820)


    asymptomatic_ihd = ExcessMortalityState('asymptomatic_ihd', disability_weight=0.08, modelable_entity_id=3233, prevalence_meid=3233)

    heart_attack_transition = IncidenceRateTransition(heart_attack, 'heart_attack', modelable_entity_id=1814)
    healthy.transition_set.append(heart_attack_transition)

    heart_failure_buckets = TransitionSet(allow_null_transition=False)
    heart_failure_buckets.extend([
        ProportionTransition(mild_heart_failure, proportion=1),
        ProportionTransition(moderate_heart_failure, proportion=1),
        ProportionTransition(severe_heart_failure, proportion=1),
        ])

    angina_buckets = TransitionSet(allow_null_transition=False)
    angina_buckets.extend([
        ProportionTransition(asymptomatic_angina, proportion=1),
        ProportionTransition(mild_angina, proportion=1),
        ProportionTransition(moderate_angina, proportion=1),
        ProportionTransition(severe_angina, proportion=1),
        ])
    healthy.transition_set.append(IncidenceRateTransition(angina_buckets, 'non_mi_angina', modelable_entity_id=1817))

    heart_attack.transition_set.allow_null_transition=False
    heart_attack.transition_set.append(ProportionTransition(heart_failure_buckets, 0.01))
    heart_attack.transition_set.append(ProportionTransition(angina_buckets, 0.15))
    heart_attack.transition_set.append(ProportionTransition(asymptomatic_ihd, 0.84))

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


def ischemic_stroke():
    module = DiseaseModule('ischemic_stroke')

    healthy = State('healthy')
    acute_1 = ExcessMortalityState('acute_ischemic_stroke_level_1', disability_weight=0.32, dwell_time=timedelta(days=28), modelable_entity_id=1827)
    acute_2 = ExcessMortalityState('acute_ischemic_stroke_level_2', disability_weight=0.32, dwell_time=timedelta(days=28), modelable_entity_id=1828)
    acute_3 = ExcessMortalityState('acute_ischemic_stroke_level_3', disability_weight=0.32, dwell_time=timedelta(days=28), modelable_entity_id=1829)
    acute_4 = ExcessMortalityState('acute_ischemic_stroke_level_4', disability_weight=0.32, dwell_time=timedelta(days=28), modelable_entity_id=1830)
    acute_5 = ExcessMortalityState('acute_ischemic_stroke_level_5', disability_weight=0.32, dwell_time=timedelta(days=28), modelable_entity_id=1831)

    acute_buckets = TransitionSet(allow_null_transition=False)
    acute_buckets.extend([
        ProportionTransition(acute_1, proportion=1),
        ProportionTransition(acute_2, proportion=1),
        ProportionTransition(acute_3, proportion=1),
        ProportionTransition(acute_4, proportion=1),
        ProportionTransition(acute_5, proportion=1),
        ])


    chronic_1 = ExcessMortalityState('chronic_ischemic_stroke_level_1', disability_weight=0.32, modelable_entity_id=1833)
    chronic_2 = ExcessMortalityState('chronic_ischemic_stroke_level_2', disability_weight=0.32, modelable_entity_id=1834)
    chronic_3 = ExcessMortalityState('chronic_ischemic_stroke_level_3', disability_weight=0.32, modelable_entity_id=1835)
    chronic_4 = ExcessMortalityState('chronic_ischemic_stroke_level_4', disability_weight=0.32, modelable_entity_id=1836)
    chronic_5 = ExcessMortalityState('chronic_ischemic_stroke_level_5', disability_weight=0.32, modelable_entity_id=1837)
    chronic_asymp = ExcessMortalityState('chronic_ischemic_stroke_asymptomatic', disability_weight=0.0, modelable_entity_id=3095)

    chronic_buckets = TransitionSet(allow_null_transition=False)
    chronic_buckets.extend([
        ProportionTransition(chronic_1, proportion=1),
        ProportionTransition(chronic_2, proportion=1),
        ProportionTransition(chronic_3, proportion=1),
        ProportionTransition(chronic_4, proportion=1),
        ProportionTransition(chronic_5, proportion=1),
        ProportionTransition(chronic_asymp, proportion=1),
        ])

    for acute in [acute_1, acute_2, acute_3, acute_4, acute_5]:
        acute.transition_set = chronic_buckets

    event_transition = IncidenceRateTransition(acute_buckets, 'ischemic_stroke', modelable_entity_id=9310)
    healthy.transition_set.append(event_transition)

    for chronic in [chronic_1, chronic_2, chronic_3, chronic_4, chronic_5, chronic_asymp]:
        chronic.transition_set.append(event_transition)

    module.states.extend([healthy, acute_1, acute_2, acute_3, acute_4, acute_5, chronic_1, chronic_2, chronic_3, chronic_4, chronic_5, chronic_asymp])

    return module

def hemorrhagic_stroke():
    module = DiseaseModule('hemorrhagic_stroke')

    healthy = State('healthy')

    acute_1 = ExcessMortalityState('acute_hemorrhagic_stroke_level_1', disability_weight=0.019, dwell_time=timedelta(days=28), modelable_entity_id=1827)
    acute_2 = ExcessMortalityState('acute_hemorrhagic_stroke_level_2', disability_weight=0.07, dwell_time=timedelta(days=28), modelable_entity_id=1828)
    acute_3 = ExcessMortalityState('acute_hemorrhagic_stroke_level_3', disability_weight=0.32, dwell_time=timedelta(days=28), modelable_entity_id=1829)
    acute_4 = ExcessMortalityState('acute_hemorrhagic_stroke_level_4', disability_weight=0.55, dwell_time=timedelta(days=28), modelable_entity_id=1830)
    acute_5 = ExcessMortalityState('acute_hemorrhagic_stroke_level_5', disability_weight=0.59, dwell_time=timedelta(days=28), modelable_entity_id=1831)

    acute_buckets = TransitionSet(allow_null_transition=False)
    acute_buckets.extend([
        ProportionTransition(acute_1, proportion=1),
        ProportionTransition(acute_2, proportion=1),
        ProportionTransition(acute_3, proportion=1),
        ProportionTransition(acute_4, proportion=1),
        ProportionTransition(acute_5, proportion=1),
        ])

    chronic_1 = ExcessMortalityState('chronic_hemorrhagic_stroke_level_1', disability_weight=0.019, modelable_entity_id=1845)
    chronic_2 = ExcessMortalityState('chronic_hemorrhagic_stroke_level_2', disability_weight=0.07, modelable_entity_id=1846)
    chronic_3 = ExcessMortalityState('chronic_hemorrhagic_stroke_level_3', disability_weight=0.316, modelable_entity_id=1847)
    chronic_4 = ExcessMortalityState('chronic_hemorrhagic_stroke_level_4', disability_weight=0.552, modelable_entity_id=1848)
    chronic_5 = ExcessMortalityState('chronic_hemorrhagic_stroke_level_5', disability_weight=0.588, modelable_entity_id=1849)
    chronic_asymp = ExcessMortalityState('chronic_hemorrhagic_stroke_asymptomatic', disability_weight=0.0, modelable_entity_id=3096)

    chronic_buckets = TransitionSet(allow_null_transition=False)
    chronic_buckets.extend([
        ProportionTransition(chronic_1, proportion=1),
        ProportionTransition(chronic_2, proportion=1),
        ProportionTransition(chronic_3, proportion=1),
        ProportionTransition(chronic_4, proportion=1),
        ProportionTransition(chronic_5, proportion=1),
        ProportionTransition(chronic_asymp, proportion=1),
        ])

    for acute in [acute_1, acute_2, acute_3, acute_4, acute_5]:
        acute.transition_set = chronic_buckets

    event_transition = IncidenceRateTransition(acute_buckets, 'hemorrhagic_stroke', modelable_entity_id=9311)
    healthy.transition_set.append(event_transition)

    for chronic in [chronic_1, chronic_2, chronic_3, chronic_4, chronic_5, chronic_asymp]:
        chronic.transition_set.append(event_transition)

    module.states.extend([healthy, acute_1, acute_2, acute_3, acute_4, acute_5, chronic_1, chronic_2, chronic_3, chronic_4, chronic_5, chronic_asymp])

    return module
