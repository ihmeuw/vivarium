# ~/ceam/tests/test_util.py

from unittest import TestCase
from datetime import timedelta
try:
    from unittest.mock import Mock
except ImportError:
    # python2 
    from mock import Mock

import numpy as np
import pandas as pd

from ceam.engine import SimulationModule
from ceam.util import sort_modules, from_yearly_rate, to_yearly_rate, rate_to_probability, probability_to_rate, only_living


class AModule(SimulationModule):
    pass
class BModule(SimulationModule):
    DEPENDENCIES = (AModule,)
class CModule(SimulationModule):
    DEPENDENCIES = (BModule,)
class DModule(SimulationModule):
    DEPENDENCIES = (CModule, BModule)


class TestSortModules(TestCase):
    # TODO: this test is not complete. There are situations in practice where the sort is wrong but this passes.
    def test_basic_function(self):
        modules = {DModule: DModule(), CModule: CModule(), BModule:BModule(), AModule:AModule()}
        sorted_modules = sort_modules(modules.values(), modules)
        self.assertListEqual(sorted_modules, [modules[AModule], modules[BModule], modules[CModule], modules[DModule]])

class TestEventListenerDecorators(TestCase):
    def test_only_living(self):
        simulation = Mock(population=pd.DataFrame({'alive': [True, True, False, True]}))

        @only_living
        def inner(label, mask, simulation):
            self.assertListEqual(list(mask), [True, True, False, True])
        inner(None, [True, True, True, True], simulation)

class TestRateConversions(TestCase):
    """
    Simple regression tests for rate functions
    """
    def test_from_yearly_rate(self):
        one_month = timedelta(days=30.5)
        rate = 0.01
        new_rate = from_yearly_rate(rate, one_month)
        self.assertAlmostEqual(new_rate, 0.0008356164383561645)

    def test_to_yearly_rate(self):
        one_month = timedelta(days=30.5)
        rate = 0.0008356164383561645
        new_rate = to_yearly_rate(rate, one_month)
        self.assertAlmostEqual(new_rate, 0.01)

    def test_rate_to_probability(self):
        rate = 0.001
        prob = rate_to_probability(rate)
        self.assertAlmostEqual(prob, 0.00099950016662497809)

    def test_probablity_to_rate(self):
        prob = 0.00099950016662497809
        rate = probability_to_rate(prob)
        self.assertAlmostEqual(rate, 0.001)

    def test_rate_to_probability_symmetry(self):
        rate = 0.0001
        for _ in range(100):
            prob = rate_to_probability(rate)
            self.assertAlmostEqual(rate, probability_to_rate(prob))
            rate += (1-0.0001)/100.0

    def test_rate_to_probablity_vectorizability(self):
        rate = 0.001
        rate = np.array([rate]*100)
        prob = rate_to_probability(rate)
        self.assertAlmostEqual(prob[10], 0.00099950016662497809)
        self.assertAlmostEqual(np.sum(rate), np.sum(probability_to_rate(prob)))


# End.
