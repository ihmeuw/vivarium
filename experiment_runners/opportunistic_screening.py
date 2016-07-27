# ~/ceam/experiment_runners/opportunistic_screening.py

import os.path
from datetime import datetime, timedelta
from time import time
from collections import defaultdict
import argparse

import pandas as pd
import numpy as np

from ceam.util import draw_count
from ceam.engine import Simulation, SimulationModule
from ceam.events import only_living, ConfigurationEvent
from ceam.modules.disease_models import heart_disease_factory, hemorrhagic_stroke_factory
from ceam.modules.healthcare_access import HealthcareAccessModule
from ceam.modules.blood_pressure import BloodPressureModule
from ceam.modules.smoking import SmokingModule
from ceam.modules.metrics import MetricsModule
from ceam.modules.sample_history import SampleHistoryModule
from ceam.modules.opportunistic_screening import OpportunisticScreeningModule, MEDICATIONS

from ceam.analysis import analyze_results, dump_results


def make_hist(start, stop, step, name, data):
    data = data[~data.isnull()]
    bins = [-float('inf')] + list(range(start, stop, step)) + [float('inf')]
    names = ['%s_lt_%s'%(name, start)] + ['%s_%d_to_%d'%(name, i, i+step) for i in bins[1:-2]] + ['%s_gte_%d'%(name, stop-step)]
    return zip(names, np.histogram(data, bins)[0])


def run_comparisons(simulation, test_modules, runs=10, verbose=False, seed=100):
    def sequences(metrics):
        dalys = [m['ylls'] + m['ylds'] for m in metrics]
        cost = [m['cost'] for m in metrics]
        ihd_counts = [m['ihd_count'] for m in metrics]
        hemorrhagic_stroke_counts = [m['hemorrhagic_stroke_count'] for m in metrics]
        return dalys, cost, ihd_counts, hemorrhagic_stroke_counts
    all_metrics = []
    for run in range(runs):
        for intervention in [True, False]:
            if intervention:
                test_modules[0].active = True
                #simulation.add_children(test_modules)
            else:
                test_modules[0].active = False
                #simulation.remove_children(test_modules)

            simulation.emit_event(ConfigurationEvent('configure_run', {'run_number': run, 'tests_active': intervention}))
            start = time()
            metrics = {}
            metrics['population_size'] = len(simulation.population)
            metrics['males'] = (simulation.population.sex == 1).sum()
            metrics['females'] = (simulation.population.sex == 2).sum()
            for name, count in make_hist(10, 110, 10, 'ages', simulation.population.age):
                metrics[name] = count

            np.random.seed(seed)
            simulation.run(datetime(1990, 1, 1), datetime(2010, 12, 1), timedelta(days=30.5))
            metrics['draw_count'] = draw_count[0]
            draw_count[0] = 0
            for m in simulation.modules:
                if isinstance(m, MetricsModule):
                    metrics.update(m.metrics)
                    break
            metrics['ihd_count'] = sum(simulation.population.ihd != 'healthy')
            metrics['heart_attack_event_count'] = simulation.population.heart_attack_event_count.sum()
            metrics['hemorrhagic_stroke_count'] = sum(simulation.population.hemorrhagic_stroke != 'healthy')
            metrics['hemorrhagic_stroke_count_event_count'] = simulation.population.hemorrhagic_stroke_event_count.sum()

            for name, count in make_hist(110, 180, 10, 'sbp', simulation.population.systolic_blood_pressure):
                metrics[name] = count

            for i, medication in enumerate(MEDICATIONS):
                if intervention:
                    metrics[medication['name']] = (simulation.population.medication_count > i).sum()
                else:
                    metrics[medication['name']] = 0

            for m in simulation.modules:
                if isinstance(m, HealthcareAccessModule):
                    metrics['healthcare_access_cost'] = sum(m.cost_by_year.values())
            if intervention:
                metrics['intervention_cost'] = sum(test_modules[0].cost_by_year.values())
                metrics['intervention'] = True
                metrics['treated_individuals'] = (simulation.population.medication_count > 0).sum()
            else:
                metrics['intervention_cost'] = 0.0
                metrics['intervention'] = False
            all_metrics.append(metrics)
            metrics['duration'] = time()-start
            simulation.reset()
            if verbose:
                print('RUN:',run)
                analyze_results(all_metrics)
    return pd.DataFrame(all_metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=1, help='Number of simulation runs to complete')
    parser.add_argument('-n', type=int, default=1, help='Instance number for this process')
    parser.add_argument('-v', action='store_true', help='Verbose logging')
    parser.add_argument('--seed', type=int, default=100, help='Random seed to use for each run')
    parser.add_argument('--stats_path', type=str, default=None, help='Output file directory. No file is written if this argument is missing')
    parser.add_argument('--detailed_sample_size', type=int, default=0, help='Number of simulants to track at highest level of detail. Resulting data will be writtent to --stats_path (or /tmp if --stats_path is ommited) as history_{instance_number}.hdf Within the hdf the group identifier will be {run number}/{True|False indicating whether the test modules were active')
    args = parser.parse_args()

    np.random.seed(args.seed)
    simulation = Simulation()

    screening_module = OpportunisticScreeningModule()
    modules = [
            screening_module,
            heart_disease_factory(),
            hemorrhagic_stroke_factory(),
            HealthcareAccessModule(),
            BloodPressureModule(),
            SmokingModule(),
            ]

    if args.detailed_sample_size:
        sample_path = args.stats_path if args.stats_path else '/tmp'
        sample_path = os.path.join(sample_path, 'history_{0}.hdf'.format(args.n))
        modules.append(SampleHistoryModule(args.detailed_sample_size, sample_path))

    metrics_module = MetricsModule()
    modules.append(metrics_module)
    for module in modules:
        module.setup()
    simulation.add_children(modules)

    simulation.load_population()
    simulation.load_data()

    results = run_comparisons(simulation, [screening_module], runs=args.runs, verbose=args.v, seed=args.seed)

    if args.stats_path:
        dump_results(results, os.path.join(args.stats_path, '%d_stats'%args.n))

    if not args.v:
        analyze_results(results)


if __name__ == '__main__':
    main()


# End.
