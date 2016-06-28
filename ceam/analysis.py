# ~/ceam/ceam/analysis.py

import argparse

import pandas as pd
import numpy as np

def confidence(seq):
    mean = np.mean(seq)
    std = np.std(seq)
    n = len(seq)
    interval = (1.96*std)/np.sqrt(n)
    return int(mean), int(mean-interval), int(mean+interval)

def difference_with_confidence(a, b):
    mean_diff = np.mean(a) - np.mean(b)
    interval = 1.96*np.sqrt(np.std(a)**2/len(a)+np.std(b)**2/len(b))
    return int(mean_diff), int(mean_diff-interval), int(mean_diff+interval)

def analyze_results(results):
    intervention = results[results.intervention == True]
    non_intervention = results[results.intervention == False]

    i_dalys = intervention.ylds + intervention.ylls
    ni_dalys = non_intervention.ylds + non_intervention.ylls

    cpd = sorted(intervention.intervention_cost.values/(ni_dalys.values-i_dalys.values))
    cpd_lower = int(cpd[int(len(cpd)*0.025)])
    cpd_upper = int(cpd[int(len(cpd)*0.975)])

    if 'qualys' in non_intervention:
        cpq = sorted(intervention.intervention_cost.values/(intervention.qualys.values - non_intervention.qualys.values))
        cpq_lower = int(cpq[int(len(cpq)*0.025)])
        cpq_upper = int(cpq[int(len(cpq)*0.975)])



    print('Total runs', len(intervention))
    print('Mean duration', int(results.duration.mean()))
    print('DALYs (intervention)', confidence(i_dalys), 'DALYs (non-intervention)', confidence(ni_dalys))
    print('DALYs averted', difference_with_confidence(ni_dalys,i_dalys))
    if 'qualys' in non_intervention:
        print('QUALYs gained', difference_with_confidence(intervention.qualys, non_intervention.qualys))
    print('Total Intervention Cost', confidence(intervention.intervention_cost))
    print('Cost per DALY averted', (int(np.mean(cpd)), cpd_lower, cpd_upper))
    if 'qualys' in non_intervention:
        print('Cost per QUALY gained', (int(np.mean(cpq)), cpq_lower, cpq_upper))
    print('IHD Count (intervention)',confidence(intervention.ihd_count), 'IHD Count (non-intervention)', confidence(non_intervention.ihd_count))
    print('Stroke Count (intervention)',confidence(intervention.hemorrhagic_stroke_count), 'Stroke Count (non-intervention)', confidence(non_intervention.hemorrhagic_stroke_count))

    print('Healthcare Access Events per year (intervention):', confidence((intervention.general_healthcare_access+intervention.followup_healthcare_access)/20))
    print('Healthcare Access Events per year (non-intervention):', confidence((non_intervention.general_healthcare_access+non_intervention.followup_healthcare_access)/20))


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
