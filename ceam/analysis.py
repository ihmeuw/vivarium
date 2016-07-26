# ~/ceam/ceam/analysis.py

import argparse

import pandas as pd
import numpy as np


def digits(x):
    if str(x) == 'nan':
        return 0
    if x == 0:
        return 1
    return int(np.ceil(np.log10(np.absolute(x))))

def round_to_1(x):
    return np.round(x, -(digits(x)-1))

def round_to_2(x):
    return np.round(x, -(digits(x)-2))
def round_to_4(x):
    return np.round(x, -(digits(x)-4))

def confidence(seq):
    t = pd.Series(seq)
    t = t.describe(percentiles=[0.025, 0.975])
    return '{:>10.0f} ({:.0f}, {:.0f})'.format(round_to_4(t['mean']), round_to_4(t['2.5%']), round_to_4(t['97.5%']))

def difference_with_confidence(a, b):
    mean_diff = np.mean(a) - np.mean(b)
    interval = 1.96*np.sqrt(np.std(a)**2/len(a)+np.std(b)**2/len(b))
    return int(mean_diff), int(mean_diff-interval), int(mean_diff+interval)


def analyze_results(results):
    intervention = results[results.intervention == True]
    non_intervention = results[results.intervention == False]

    i_dalys = intervention.ylds + intervention.ylls
    ni_dalys = non_intervention.ylds + non_intervention.ylls

    i_dalys.index = range(len(i_dalys))
    ni_dalys.index = range(len(i_dalys))
    non_intervention.intervention_cost.index = range(len(i_dalys))
    intervention.intervention_cost.index = range(len(i_dalys))

    print(
"""
 Total runs: {runs}
Min runtime: {duration:.0f} seconds
""".format(runs=len(intervention),
           duration=results.duration.min()))
    print('       DALYs (non-intervention):', confidence(ni_dalys))
    print('           DALYs (intervention):', confidence(i_dalys))
    print('                  DALYs averted:', confidence(ni_dalys - i_dalys))
    print('  Total Cost (non-intervention):', confidence(non_intervention.intervention_cost))
    print('      Total Cost (intervention):', confidence(intervention.intervention_cost))
    print('                 Change in Cost:', confidence(intervention.intervention_cost - non_intervention.intervention_cost))
    print('          Cost per DALY averted:', confidence((intervention.intervention_cost - non_intervention.intervention_cost)/(ni_dalys - i_dalys)))
    print()
    print('   IHD Count (non-intervention):', confidence(non_intervention.ihd_count))
    print('       IHD Count (intervention):', confidence(intervention.ihd_count))
    print('Stroke Count (non-intervention):', confidence(non_intervention.hemorrhagic_stroke_count))
    print('    Stroke Count (intervention):', confidence(intervention.hemorrhagic_stroke_count))
    print()
    print('Healthcare per year (non-intervention):', confidence((non_intervention.general_healthcare_access+non_intervention.followup_healthcare_access)/20))
    print('                               General:', confidence((non_intervention.general_healthcare_access)/20))
    print('                              Followup:', confidence((non_intervention.followup_healthcare_access)/20))
    print('    Healthcare per year (intervention):', confidence((intervention.general_healthcare_access+intervention.followup_healthcare_access)/20))
    print('                               General:', confidence((intervention.general_healthcare_access)/20))
    print('                              Followup:', confidence((intervention.followup_healthcare_access)/20))
    print('    Treated Individuals (intervention):', confidence(intervention.treated_individuals))


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
