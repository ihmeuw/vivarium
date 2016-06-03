# ~/ceam/experiment_runners/level_3.py

import numpy as np
from time import time
from datetime import datetime, timedelta

from ceam.engine import Simulation
from ceam.modules.ihd import IHDModule
from ceam.modules.hemorrhagic_stroke import HemorrhagicStrokeModule
from ceam.modules.level3intervention import Level3InterventionModule
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
            simulation.run(datetime(1990, 1, 1), datetime(2013, 12, 31), timedelta(days=30.5)) #TODO: Is 30.5 days a good enough approximation of one month? -Alec
            metrics = dict(simulation._modules[MetricsModule].metrics)
            metrics['ihd_count'] = sum(simulation.population.ihd == True)
            metrics['hemorrhagic_stroke_count'] = sum(simulation.population.hemorrhagic_stroke == True)
            if do_test:
                metrics['cost'] = test_modules[0].cummulative_cost
                test_a_metrics.append(metrics)
            else:
                metrics['cost'] = 0.0
                test_b_metrics.append(metrics)
            print()
            print('Duration: %s'%(time()-start)) # fix
            print()
            simulation.reset()

        a_dalys, a_cost, a_ihd_counts, a_hemorrhagic_stroke_counts = sequences(test_a_metrics)
        b_dalys, b_cost, b_ihd_counts, b_hemorrhagic_stroke_counts = sequences(test_b_metrics)
        per_daly = [cost/(b-a) for a,b,cost in zip(a_dalys, b_dalys, a_cost)]
        print()
        print("IHD count (without intervention):       ", b_ihd_counts)
        print("IHD count (with intervention):          ", a_ihd_counts)
        print("Hem Stroke count (without intervention):", b_hemorrhagic_stroke_counts)
        print("Hem Stroke count (with intervention):   ", a_hemorrhagic_stroke_counts)
        print("IHD count averted:                      ", difference_with_confidence(b_ihd_counts, a_ihd_counts))
        print("Hem Strokes averted:                    ", difference_with_confidence(b_hemorrhagic_stroke_counts, a_hemorrhagic_stroke_counts))
        print("DALYs averted:                          ", difference_with_confidence(b_dalys, a_dalys))
        print("Total cost:                             ", confidence(a_cost))
        print("Cost per DALY:                          ", confidence(per_daly))


def main():
    simulation = Simulation()

    modules = [BloodPressureModule(), IHDModule(), HemorrhagicStrokeModule(), MetricsModule()]
    screening_module = Level3InterventionModule()
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
