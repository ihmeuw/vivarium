from unittest import TestCase

from engine import SimulationModule
from util import sort_modules

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
