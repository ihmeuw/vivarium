# ~/ceam/experiment_runners/level_3.py

from __future__ import print_function                                   # So 'print("Foo:", 5)' prints "Foo: 5" in both Python2 and Python3.

import numpy as np
from time import time
from datetime import datetime, timedelta

from ceam.engine import Simulation
from ceam.modules.chronic_condition import ChronicConditionModule
from ceam.modules.level3intervention_scalarized import Level3InterventionScalarizedModule
from ceam.modules.blood_pressure import BloodPressureModule
from ceam.modules.metrics import MetricsModule


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
            simulation.run(datetime(1990, 1, 1), datetime(2010, 12, 31), timedelta(days=30.5)) #TODO: Is 30.5 days a good enough approximation of one month? -Alec
            metrics = dict(simulation._modules[MetricsModule].metrics)
            metrics['ihd_count'] = sum(simulation.population.ihd == True)
            metrics['hemorrhagic_stroke_count'] = sum(simulation.population.hemorrhagic_stroke == True)
            if do_test:
                metrics['cost'] = test_modules[0].cummulative_cost
                test_a_metrics.append(metrics)
            else:
                metrics['cost'] = 0.0
                test_b_metrics.append(metrics)
            print("")
            print('Duration of this run: %s seconds.'%(time() - start))
            simulation.reset()

        a_dalys, a_cost, a_ihd_counts, a_hemorrhagic_stroke_counts = sequences(test_a_metrics)
        b_dalys, b_cost, b_ihd_counts, b_hemorrhagic_stroke_counts = sequences(test_b_metrics)
        cost_per_daly_averted = [cost/(b-a) for a,b,cost in zip(a_dalys, b_dalys, a_cost)]
        print("")
        print("IHD count (without intervention):       ", b_ihd_counts)
        print("IHD count (with intervention):          ", a_ihd_counts)
        print("Hem Stroke count (without intervention):", b_hemorrhagic_stroke_counts)
        print("Hem Stroke count (with intervention):   ", a_hemorrhagic_stroke_counts)
        print("IHD count averted:                      ", difference_with_confidence(b_ihd_counts, a_ihd_counts))
        print("Hem Strokes averted:                    ", difference_with_confidence(b_hemorrhagic_stroke_counts, a_hemorrhagic_stroke_counts))
        print("DALYs averted:                          ", difference_with_confidence(b_dalys, a_dalys))
        print("Total cost:                             ", confidence(a_cost))
        print("Cost per DALY averted:                  ", confidence(cost_per_daly_averted))


def main():
    total_time_start = time()

    simulation = Simulation()

    modules = [
               ChronicConditionModule('ihd', 'ihd_mortality_rate.csv', 'IHD incidence rates.csv', 0.08),
               ChronicConditionModule('hemorrhagic_stroke', 'chronic_hem_stroke_excess_mortality.csv', 'hem_stroke_incidence_rates.csv', 0.316),
               MetricsModule(),
              ]
    screening_module = Level3InterventionScalarizedModule()
    modules.append(screening_module)
    for module in modules:
        module.setup()
    simulation.register_modules(modules)

    simulation.load_population()
    simulation.load_data()

    run_comparisons(simulation, [screening_module], runs=10)

    print("")
    print('Total time for entire simulation: %s seconds.'%(time() - total_time_start))
    print("")


if __name__ == '__main__':
    main()


# End.
