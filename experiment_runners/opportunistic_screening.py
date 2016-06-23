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
from ceam.modules.opportunistic_screening import OpportunisticScreeningModule, MEDICATIONS

from ceam.analysis import analyze_results, dump_results


def make_hist(start, stop, step, name, data):
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
                # TODO: This is a hack to fix the bug in the module sorting code until Bob can get a general fix out
                simulation._ordered_modules.remove(test_modules[0])
                simulation._ordered_modules.append(test_modules[0])
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

    screening_module = OpportunisticScreeningModule()
    modules = [
            screening_module,
            ChronicConditionModule('ihd', 'ihd_mortality_rate.csv', 'ihd_incidence_rates.csv', 0.08, acute_mortality_table_name='mi_acute_excess_mortality.csv'),
            ChronicConditionModule('hemorrhagic_stroke', 'chronic_hem_stroke_excess_mortality.csv', 'hem_stroke_incidence_rates.csv', 0.316, acute_mortality_table_name='acute_hem_stroke_excess_mortality.csv'),
            HealthcareAccessModule(),
            BloodPressureModule(),
            SmokingModule(),
            ]
    metrics_module = MetricsModule()
    modules.append(metrics_module)
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
