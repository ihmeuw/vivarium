# ~/ceam/experiment_runners/opportunistic_screening.py

from datetime import datetime, timedelta
from time import time
from collections import defaultdict

import pandas as pd
import numpy as np

from ceam.engine import Simulation, SimulationModule
from ceam.util import only_living
from ceam.modules.ihd import IHDModule
from ceam.modules.hemorrhagic_stroke import HemorrhagicStrokeModule
from ceam.modules.healthcare_access import HealthcareAccessModule
from ceam.modules.blood_pressure import BloodPressureModule
from ceam.modules.metrics import MetricsModule


pd.set_option('mode.chained_assignment', 'raise')


def _hypertensive_categories(population):
        under_60 = population.age < 60
        over_60 = population.age >= 60
        under_140 = population.systolic_blood_pressure < 140
        under_150 = population.systolic_blood_pressure < 150
        under_180 = population.systolic_blood_pressure < 180

        normotensive = under_60 & (under_140)
        normotensive |= over_60 & (under_150)

        hypertensive = under_60 & (~under_140) & (under_180)
        hypertensive |= over_60 & (~under_150) & (under_180)

        severe_hypertension = (~under_180)

        return (population.loc[normotensive], population.loc[hypertensive], population.loc[severe_hypertension])


class OpportunisticScreeningModule(SimulationModule):
    DEPENDS = (BloodPressureModule, HealthcareAccessModule,)

    def setup(self):
        self.cost_by_year = defaultdict(int)
        self.register_event_listener(self.non_followup_blood_pressure_test, 'general_healthcare_access')
        self.register_event_listener(self.followup_blood_pressure_test, 'followup_healthcare_access')
        self.register_event_listener(self.track_monthly_cost, 'time_step')
        self.register_event_listener(self.adjust_blood_pressure, 'time_step')

    def load_population_columns(self, path_prefix, population_size):
        #TODO: Some people will start out taking medications?
        self.population_columns = pd.DataFrame({
            'taking_blood_pressure_medication_a': [False]*population_size,
            'taking_blood_pressure_medication_b': [False]*population_size,
            })

    def non_followup_blood_pressure_test(self, event):
        self.cost_by_year[self.simulation.current_time.year] += len(event.affected_population) * 3.0

        #TODO: testing error

        normotensive, hypertensive, severe_hypertension = _hypertensive_categories(event.affected_population)

        # Normotensive simulants get a 60 month followup and no drugs
        self.simulation.population.loc[normotensive.index, 'healthcare_followup_date'] = self.simulation.current_time + timedelta(days= 30.5*60) # 60 months

        # Hypertensive simulants get a 1 month followup and no drugs
        self.simulation.population.loc[hypertensive.index, 'healthcare_followup_date'] = self.simulation.current_time + timedelta(days= 30.5) # 1 month

        # Severe hypertensive simulants get a 1 month followup and all drugs
        self.simulation.population.loc[severe_hypertension.index, 'healthcare_followup_date'] = self.simulation.current_time + timedelta(days= 30.5*6) # 6 months
        self.simulation.population.loc[severe_hypertension.index, 'taking_blood_pressure_medication_a'] = True
        self.simulation.population.loc[severe_hypertension.index, 'taking_blood_pressure_medication_b'] = True

    def followup_blood_pressure_test(self, event):
        self.cost_by_year[self.simulation.current_time.year] += len(event.affected_population) * 3.0

        normotensive, hypertensive, severe_hypertension = _hypertensive_categories(event.affected_population)

        nonmedicated_normotensive = normotensive.loc[(normotensive.taking_blood_pressure_medication_a == False) & (normotensive.taking_blood_pressure_medication_b == False)]
        medicated_normotensive = normotensive.loc[(normotensive.taking_blood_pressure_medication_a == False) | (normotensive.taking_blood_pressure_medication_b == False)]

        # Unmedicated normotensive simulants get a 60 month followup
        self.simulation.population.loc[nonmedicated_normotensive.index, 'healthcare_followup_date'] = self.simulation.current_time + timedelta(days= 30.5*60) # 60 months

        # Medicated normotensive simulants drop their drugs and get an 11 month followup
        self.simulation.population.loc[medicated_normotensive.index, 'healthcare_followup_date'] = self.simulation.current_time + timedelta(days= 30.5*11) # 11 months
        self.simulation.population.loc[medicated_normotensive.index, 'taking_blood_pressure_medication_a'] = False
        self.simulation.population.loc[medicated_normotensive.index, 'taking_blood_pressure_medication_b'] = False

        # Hypertensive simulants get a 6 month followup and go on one drug
        # TODO: what if they are already taking drugs?
        self.simulation.population.loc[hypertensive.index, 'healthcare_followup_date'] = self.simulation.current_time + timedelta(days= 30.5*6) # 6 months
        self.simulation.population.loc[hypertensive.index, 'taking_blood_pressure_medication_a'] = True

        # Severe hypertensive simulants get the same treatment as during a non-followup test
        # TODO: is this right?
        self.simulation.population.loc[severe_hypertension.index, 'healthcare_followup_date'] = self.simulation.current_time + timedelta(days= 30.5*6) # 6 months
        self.simulation.population.loc[severe_hypertension.index, 'taking_blood_pressure_medication_a'] = True
        self.simulation.population.loc[severe_hypertension.index, 'taking_blood_pressure_medication_b'] = True

    @only_living
    def track_monthly_cost(self, event):
        #TODO: realistic costs
        for medication in ['medication_a', 'medication_b']:
            medication_cost = self.simulation.config.getfloat('opportunistic_screening', medication + '_cost')
            medication_cost *= self.simulation.config.getfloat('opportunistic_screening', 'adherence')
            self.cost_by_year[self.simulation.current_time.year] += (event.affected_population['taking_blood_pressure_'+medication] == True).sum() * medication_cost*self.simulation.last_time_step.days

    @only_living
    def adjust_blood_pressure(self, event):
        # TODO: Real drug effects + adherance rates
        for medication in ['medication_a', 'medication_b']:
            medication_effect = self.simulation.config.getfloat('opportunistic_screening', medication + '_effectiveness')
            medication_effect *= self.simulation.config.getfloat('opportunistic_screening', 'adherence')
            affected_population = event.affected_population.loc[event.affected_population['taking_blood_pressure_'+medication] == True]
            self.simulation.population.loc[event.affected_population.index, 'systolic_blood_pressure'] -= medication_effect


def confidence(seq):
    mean = np.mean(seq)
    std = np.std(seq)
    runs = len(seq)
    interval = (1.96*std)/np.sqrt(runs)
    return mean, mean-interval, mean+interval

def difference_with_confidence(a, b):
    mean_diff = np.mean(a) - np.mean(b)
    interval = 1.96*np.sqrt(np.std(a)/len(a)+np.std(b)/len(b))
    return mean_diff, int(mean_diff-interval), int(mean_diff+interval)

def run_comparisons(simulation, test_modules, runs=10):
    def sequences(metrics):
        dalys = [m['ylls'] + m['ylds'] for m in metrics]
        cost = [m['cost'] for m in metrics]
        ihd_counts = [m['ihd_count'] for m in metrics]
        hemorrhagic_stroke_counts = [m['hemorrhagic_stroke_count'] for m in metrics]
        return dalys, cost, ihd_counts, hemorrhagic_stroke_counts
    test_a_metrics = []
    test_b_metrics = []
    for run in range(runs):
        for do_test in [True, False]:
            if do_test:
                simulation.register_modules(test_modules)
            else:
                simulation.deregister_modules(test_modules)

            start = time()
            simulation.run(datetime(1990, 1, 1), datetime(2013, 12, 31), timedelta(days=30.5)) #TODO: Is 30.5 days a good enough approximation of one month? -Alec
            metrics = dict(simulation._modules[MetricsModule].metrics)
            metrics['ihd_count'] = sum(simulation.population.ihd == True)
            metrics['hemorrhagic_stroke_count'] = sum(simulation.population.hemorrhagic_stroke == True)
            metrics['sbp'] = np.mean(simulation.population.systolic_blood_pressure)
            if do_test:
                metrics['cost'] = sum(test_modules[0].cost_by_year.values())
                test_a_metrics.append(metrics)
            else:
                metrics['cost'] = 0.0
                test_b_metrics.append(metrics)
            print('Duration: %s'%(time()-start))
            simulation.reset()

        a_dalys, a_cost, a_ihd_counts, a_hemorrhagic_stroke_counts = sequences(test_a_metrics)
        b_dalys, b_cost, b_ihd_counts, b_hemorrhagic_stroke_counts = sequences(test_b_metrics)
        per_daly = [(b-a)/cost for a,b,cost in zip(a_dalys, b_dalys, a_cost)]
        print(per_daly)
        print("DALYs averted:", difference_with_confidence(b_dalys, a_dalys))
        print("Total cost:", confidence(a_cost))
        print("Cost per DALY:", confidence(per_daly))


def main():
    simulation = Simulation()

    modules = [IHDModule(), HemorrhagicStrokeModule(), HealthcareAccessModule(), BloodPressureModule()]
    metrics_module = MetricsModule()
    modules.append(metrics_module)
    screening_module = OpportunisticScreeningModule()
    modules.append(screening_module)
    for module in modules:
        module.setup()
    simulation.register_modules(modules)

    simulation.load_population()
    simulation.load_data()

    run_comparisons(simulation, [screening_module], runs=10)


if __name__ == '__main__':
    main()


# End.
