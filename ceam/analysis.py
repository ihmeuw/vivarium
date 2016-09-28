# ~/ceam/ceam/analysis.py

import argparse
import os.path

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


def analyze_results_to_df(results):
    mixed_results = pd.DataFrame()
    for i, r in enumerate(results):
        mixed_results = mixed_results.append(r)

    comparisons = list(set(mixed_results.comparison.unique()) - {'base'})
    if len(comparisons) != 1:
        raise ValueError("Don't know what to do with more that one non-base comparison")

    intervention = mixed_results[mixed_results.comparison == comparisons[0]].reset_index(drop=True)
    non_intervention = mixed_results[mixed_results.comparison == 'base'].reset_index(drop=True)

    i_dalys = intervention.ylds + intervention.ylls
    ni_dalys = non_intervention.ylds + non_intervention.ylls

    i_dalys.index = range(len(i_dalys))
    ni_dalys.index = range(len(i_dalys))
    non_intervention.intervention_cost.index = range(len(i_dalys))
    intervention.intervention_cost.index = range(len(i_dalys))

    dalys_averted = []
    ylds_averted = []
    ylls_averted = []
    for r in results:
        lintervention = r[r.intervention == True].reset_index(drop=True)
        lnon_intervention = r[r.intervention == False].reset_index(drop=True)
        dalys_averted.extend((lnon_intervention.ylds + lnon_intervention.ylls) - (lintervention.ylds + lintervention.ylls))
        ylls_averted.extend((lnon_intervention.ylls) - (lintervention.ylls))
        ylds_averted.extend((lnon_intervention.ylds) - (lintervention.ylds))

    data = {
            'DALYs (non-intervention)': ni_dalys,
            'DALYs (intervention)': i_dalys,
            'DALYs averted': dalys_averted,
            'YLLs averted': ylls_averted,
            'YLDs averted': ylds_averted,
            'Total Cost (non-intervention)': non_intervention.intervention_cost,
            'Total Cost (intervention)': intervention.intervention_cost,
            'Change in Cost': intervention.intervention_cost - non_intervention.intervention_cost,
            'Cost per DALY averted': (intervention.intervention_cost - non_intervention.intervention_cost)/(dalys_averted),
            'IHD Count (non-intervention)': non_intervention.ihd_count,
            'IHD Count (intervention)': intervention.ihd_count,
            'Stroke Count (non-intervention)': non_intervention.hemorrhagic_stroke_count,
            'Stroke Count (intervention)': intervention.hemorrhagic_stroke_count,
            'Healthcare per year (non-intervention)': (non_intervention.general_healthcare_access+non_intervention.followup_healthcare_access)/20,
            'Healthcare per year (non-intervention): General': (non_intervention.general_healthcare_access)/20,
            'Healthcare per year (non-intervention): Followup': (non_intervention.followup_healthcare_access)/20,
            'Healthcare per year (intervention)': (intervention.general_healthcare_access+intervention.followup_healthcare_access)/20,
            'Healthcare per year (intervention): General': (intervention.general_healthcare_access)/20,
            'Healthcare per year (intervention): Followup': (intervention.followup_healthcare_access)/20,
            'Treated Individuals (intervention)': intervention.treated_individuals,
            }
    data = {k:[np.mean(v)] for k,v in data.items()}
    return mixed_results, pd.DataFrame(data)


def analyze_results(results):
    mixed_results = pd.DataFrame()
    for r in results:
        mixed_results = mixed_results.append(r)

    comparisons = list(set(mixed_results.comparison.unique()) - {'base'})
    if len(comparisons) != 1:
        raise ValueError("Don't know what to do with more that one non-base comparison")

    intervention = mixed_results[mixed_results.comparison == comparisons[0]].reset_index(drop=True)
    non_intervention = mixed_results[mixed_results.comparison == 'base'].reset_index(drop=True)

    i_dalys = intervention.years_lived_with_disability + intervention.years_of_life_lost
    ni_dalys = non_intervention.years_lived_with_disability + non_intervention.years_of_life_lost

    i_dalys.index = range(len(i_dalys))
    ni_dalys.index = range(len(i_dalys))

    dalys_averted = []
    ylds_averted = []
    ylls_averted = []
    for r in results:
        lintervention = r[r.comparison != 'base'].reset_index(drop=True)
        lnon_intervention = r[r.comparison == 'base'].reset_index(drop=True)
        dalys_averted.extend((lnon_intervention.years_lived_with_disability + lnon_intervention.years_of_life_lost) - (lintervention.years_lived_with_disability + lintervention.years_of_life_lost))
        ylls_averted.extend((lnon_intervention.years_of_life_lost) - (lintervention.years_of_life_lost))
        ylds_averted.extend((lnon_intervention.years_lived_with_disability) - (lintervention.years_lived_with_disability))

    print(
"""
 Total runs: {runs}
Min runtime: {duration:.0f} seconds
Mean runtime: {mean_duration} seconds
""".format(runs=len(intervention),
           duration=mixed_results.duration.min(),
           mean_duration=confidence(mixed_results.duration)))
    print('       DALYs (non-intervention):', confidence(ni_dalys))
    print('           DALYs (intervention):', confidence(i_dalys))
    print('                  DALYs averted:', confidence(dalys_averted))
    print('                  YLLs averted:', confidence(ylls_averted))
    print('                  YLDs averted:', confidence(ylds_averted))
    print('  Total Cost (non-intervention):', confidence(non_intervention.cost))
    print('      Total Cost (intervention):', confidence(intervention.cost))
    print('                 Change in Cost:', confidence(intervention.cost - non_intervention.cost))
    if np.all(pd.Series(dalys_averted) == 0):
        print('          Cost per DALY averted: NA')
    else:
        print('          Cost per DALY averted:', confidence((intervention.cost - non_intervention.cost)/(dalys_averted)))
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
    results = []
    for path in sorted(paths):
        if 'cfg' in path:
            continue
        result = pd.read_csv(path)
        result['iteration'] = int(os.path.basename(path).split('_')[1])
        results.append(result)
    return results


def main():
    import sys
    analyze_results(load_results(sys.argv[1:]))


if __name__ == '__main__':
    main()


# End.
