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
    interval = 1.96*np.sqrt(np.std(a)/len(a)+np.std(b)/len(b))
    return mean_diff, int(mean_diff-interval), int(mean_diff+interval)

def analyze_results(results):
    for col in results.columns:
        print(col, confidence(results[col]))

def dump_results(results, path):
    results.to_csv(path)

def load_results(path):
    return pd.read_csv(path)

def main():
    pass

if __name__ == '__main__':
    main()
