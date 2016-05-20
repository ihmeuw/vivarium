from unittest import TestCase
from datetime import timedelta

from ceam.engine import SimulationModule
from ceam.util import sort_modules, from_yearly_rate, to_yearly_rate

class AModule(SimulationModule):
    pass
class BModule(SimulationModule):
    DEPENDENCIES = (AModule,)
class CModule(SimulationModule):
    DEPENDENCIES = (BModule,)
class DModule(SimulationModule):
    DEPENDENCIES = (CModule, BModule)

class TestSortModules(TestCase):
    def test_basic_function(self):
        modules = {DModule: DModule(), CModule: CModule(), BModule:BModule(), AModule:AModule()}
        sorted_modules = sort_modules(modules)
        self.assertListEqual(sorted_modules, [modules[AModule], modules[BModule], modules[CModule], modules[DModule]])

class TestRateConversions(TestCase):
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

