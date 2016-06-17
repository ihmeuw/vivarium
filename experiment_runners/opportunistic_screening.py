# ~/ceam/experiment_runners/opportunistic_screening.py

import os.path
from datetime import datetime, timedelta
from time import time
from collections import defaultdict
import argparse

import pandas as pd
import numpy as np

from ceam.engine import Simulation, SimulationModule
from ceam.events import only_living
from ceam.modules.chronic_condition import ChronicConditionModule
from ceam.modules.healthcare_access import HealthcareAccessModule
from ceam.modules.blood_pressure import BloodPressureModule
from ceam.modules.smoking import SmokingModule
from ceam.modules.metrics import MetricsModule

from ceam.analysis import analyze_results, dump_results


pd.set_option('mode.chained_assignment', 'raise')

#TODO: This feels like configuration but is difficult to express in ini type files
MEDICATIONS = [
        {
            'name': 'Thiazide-type diuretics',
            'daily_cost': 0.009,
            'efficacy': 8.8,
        },
        {
            'name': 'Calcium-channel blockers',
            'daily_cost': 0.166,
            'efficacy': 8.8,
        },
        {
            'name': 'ACE Inhibitors',
            'daily_cost': 0.059,
            'efficacy': 10.3,
        },
        {
            'name': 'Beta blockers',
            'daily_cost': 0.048,
            'efficacy': 9.2,
        },
]


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
        self.population_columns['medication_count'] = [0]*population_size

    def non_followup_blood_pressure_test(self, event):
        self.cost_by_year[self.simulation.current_time.year] += len(event.affected_population) * self.simulation.config.getfloat('opportunistic_screening', 'blood_pressure_test_cost')

        #TODO: testing error

        normotensive, hypertensive, severe_hypertension = _hypertensive_categories(event.affected_population)

        # Normotensive simulants get a 60 month followup and no drugs
        self.simulation.population.loc[normotensive.index, 'healthcare_followup_date'] = self.simulation.current_time + timedelta(days= 30.5*60) # 60 months

        # Hypertensive simulants get a 1 month followup and no drugs
        self.simulation.population.loc[hypertensive.index, 'healthcare_followup_date'] = self.simulation.current_time + timedelta(days= 30.5) # 1 month

        # Severe hypertensive simulants get a 1 month followup and two drugs
        self.simulation.population.loc[severe_hypertension.index, 'healthcare_followup_date'] = self.simulation.current_time + timedelta(days= 30.5*6) # 6 months

        self.simulation.population.loc[severe_hypertension.index, 'medication_count'] = np.minimum(severe_hypertension['medication_count'] + 2, len(MEDICATIONS))

    def followup_blood_pressure_test(self, event):
        self.cost_by_year[self.simulation.current_time.year] += len(event.affected_population) * (self.simulation.config.getfloat('opportunistic_screening', 'blood_pressure_test_cost') + self.simulation.config.getfloat('appointments', 'cost'))

        normotensive, hypertensive, severe_hypertension = _hypertensive_categories(event.affected_population)

        nonmedicated_normotensive = normotensive.loc[normotensive.medication_count == 0]
        medicated_normotensive = normotensive.loc[normotensive.medication_count > 0]

        # Unmedicated normotensive simulants get a 60 month followup
        self.simulation.population.loc[nonmedicated_normotensive.index, 'healthcare_followup_date'] = self.simulation.current_time + timedelta(days= 30.5*60) # 60 months

        # Medicated normotensive simulants get an 11 month followup
        self.simulation.population.loc[medicated_normotensive.index, 'healthcare_followup_date'] = self.simulation.current_time + timedelta(days= 30.5*11) # 11 months

        # Hypertensive simulants get a 6 month followup and go on one drug
        self.simulation.population.loc[hypertensive.index, 'healthcare_followup_date'] = self.simulation.current_time + timedelta(days= 30.5*6) # 6 months
        self.simulation.population.loc[hypertensive.index, 'medication_count'] = np.minimum(hypertensive['medication_count'] + 1, len(MEDICATIONS))
        self.simulation.population.loc[severe_hypertension.index, 'healthcare_followup_date'] = self.simulation.current_time + timedelta(days= 30.5*6) # 6 months
        self.simulation.population.loc[severe_hypertension.index, 'medication_count'] = np.minimum(severe_hypertension.medication_count + 1, len(MEDICATIONS))


    @only_living
    def track_monthly_cost(self, event):
        for medication_number in range(len(MEDICATIONS)):
            user_count = (event.affected_population.medication_count > medication_number).sum()
            self.cost_by_year[self.simulation.current_time.year] += user_count * MEDICATIONS[medication_number]['daily_cost'] * self.simulation.last_time_step.days

    @only_living
    def adjust_blood_pressure(self, event):
        for medication_number in range(len(MEDICATIONS)):
            medication_efficacy = MEDICATIONS[medication_number]['efficacy'] * self.simulation.config.getfloat('opportunistic_screening', 'adherence')
            affected_population = event.affected_population[event.affected_population.medication_count > medication_number]
            self.simulation.population.loc[affected_population.index, 'systolic_blood_pressure'] -= medication_efficacy

def make_hist(start, stop, step, name, data):
    #TODO there are NANs in the systolic blood pressure data, why?

    data = data[~data.isnull()]
    bins = [-float('inf')] + list(range(start, stop, step)) + [float('inf')]
    names = ['%s_lt_%s'%(name, start)] + ['%s_%d_to_%d'%(name, i, i+step) for i in bins[1:-2]] + ['%s_gte_%d'%(name, stop-step)]
    return zip(names, np.histogram(data, bins)[0])

def run_comparisons(simulation, test_modules, runs=10, verbose=False):
    def sequences(metrics):
        dalys = [m['ylls'] + m['ylds'] for m in metrics]
        cost = [m['cost'] for m in metrics]
        ihd_counts = [m['ihd_count'] for m in metrics]
        hemorrhagic_stroke_counts = [m['hemorrhagic_stroke_count'] for m in metrics]
        return dalys, cost, ihd_counts, hemorrhagic_stroke_counts
    all_metrics = []
    for run in range(runs):
        for do_test in [True, False]:
            if do_test:
                simulation.register_modules(test_modules)
            else:
                simulation.deregister_modules(test_modules)

            start = time()
            metrics = {}
            metrics['population_size'] = len(simulation.population)
            metrics['males'] = (simulation.population.sex == 1).sum()
            metrics['females'] = (simulation.population.sex == 2).sum()
            for name, count in make_hist(10, 110, 10, 'ages', simulation.population.age):
                metrics[name] = count

            simulation.run(datetime(1990, 1, 1), datetime(2010, 12, 1), timedelta(days=30.5)) #TODO: Is 30.5 days a good enough approximation of one month? -Alec
            metrics.update(simulation._modules[MetricsModule].metrics)
            metrics['ihd_count'] = sum(simulation.population.ihd == True)
            metrics['hemorrhagic_stroke_count'] = sum(simulation.population.hemorrhagic_stroke == True)
            metrics['hemorrhagic_stroke_count'] = sum(simulation.population.hemorrhagic_stroke == True)

            for name, count in make_hist(110, 180, 10, 'sbp', simulation.population.systolic_blood_pressure):
                metrics[name] = count

            for i, medication in enumerate(MEDICATIONS):
                if do_test:
                    metrics[medication['name']] = (simulation.population.medication_count > i).sum()
                else:
                    metrics[medication['name']] = 0

            metrics['healthcare_access_cost'] = sum(simulation._modules[HealthcareAccessModule].cost_by_year.values())
            if do_test:
                metrics['intervention_cost'] = sum(test_modules[0].cost_by_year.values())
                metrics['intervention'] = True
            else:
                metrics['intervention_cost'] = 0.0
                metrics['intervention'] = False
            all_metrics.append(metrics)
            metrics['duration'] = time()-start
            simulation.reset()
            if verbose:
                print('RUN:',run)
                analyze_results(pd.DataFrame(all_metrics))
    return pd.DataFrame(all_metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=1, help='Number of simulation runs to complete')
    parser.add_argument('-n', type=int, default=1, help='Instance number for this process')
    parser.add_argument('-v', action='store_true', help='Verbose logging')
    parser.add_argument('--stats_path', type=str, default=None, help='Output file directory. No file is written if this argument is missing')
    args = parser.parse_args()

    simulation = Simulation()

    modules = [
            ChronicConditionModule('ihd', 'ihd_mortality_rate.csv', 'ihd_incidence_rates.csv', 0.08),
            ChronicConditionModule('hemorrhagic_stroke', 'chronic_hem_stroke_excess_mortality.csv', 'hem_stroke_incidence_rates.csv', 0.316),
            HealthcareAccessModule(),
            BloodPressureModule(),
            SmokingModule(),
            ]
    metrics_module = MetricsModule()
    modules.append(metrics_module)
    screening_module = OpportunisticScreeningModule()
    modules.append(screening_module)
    for module in modules:
        module.setup()
    simulation.register_modules(modules)

    simulation.load_population()
    simulation.load_data()

    results = run_comparisons(simulation, [screening_module], runs=args.runs, verbose=args.v)

    if args.stats_path:
        dump_results(results, os.path.join(args.stats_path, '%d_stats'%args.n))

    if not args.v:
        analyze_results(results)


if __name__ == '__main__':
    main()


# End.
