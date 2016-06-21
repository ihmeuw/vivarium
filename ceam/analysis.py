# ~/ceam/ceam/analysis.py

import argparse

import pandas as pd
import numpy as np

def confidence(seq):
    mean = np.mean(seq)
    std = np.std(seq)
    runs = len(seq)
    interval = (1.96*std)/np.sqrt(runs)
    return mean, mean-interval, mean+interval

def difference_with_confidence(a, b):
    mean_diff = np.mean(a) - np.mean(b)
    interval = 1.96*np.sqrt(np.std(a)**2/len(a)+np.std(b)**2/len(b))
    return mean_diff, int(mean_diff-interval), int(mean_diff+interval)

def analyze_results(results):
    intervention = results[results.intervention == True]
    non_intervention = results[results.intervention == False]

    i_dalys = intervention.ylds + intervention.ylls
    ni_dalys = non_intervention.ylds + non_intervention.ylls

    print('Total runs', len(intervention))
    print('Mean duration', results.duration.mean())
    print('DALYs (intervention)', confidence(i_dalys), 'DALYs (non-intervention)', confidence(ni_dalys))
    print('DALYs averted', difference_with_confidence(ni_dalys,i_dalys))
    print('Total Intervention Cost', confidence(intervention.intervention_cost))
    print('Cost per DALY', confidence(intervention.intervention_cost.values/(ni_dalys.values-i_dalys.values)))
    print('IHD Count (intervention)',confidence(intervention.ihd_count), 'IHD Count (non-intervention)', confidence(non_intervention.ihd_count))
    print('Stroke Count (intervention)',confidence(intervention.hemorrhagic_stroke_count), 'Stroke Count (non-intervention)', confidence(non_intervention.hemorrhagic_stroke_count))

    print('Healthcare Access Events per year (intervention):', confidence((intervention.general_healthcare_access+intervention.followup_healthcare_access)/20))
    print('Healthcare Access Events per year (non-non_intervention):', confidence((non_intervention.general_healthcare_access+non_intervention.followup_healthcare_access)/20))


def dump_results(results, path):
    results.to_csv(path)

def load_results(paths):
    results = pd.DataFrame()
    for path in paths:
        results = results.append(pd.read_csv(path))
    return results

def main():
    import sys
    analyze_results(load_results(sys.argv[1:]))


if __name__ == '__main__':
    main()


# End.
