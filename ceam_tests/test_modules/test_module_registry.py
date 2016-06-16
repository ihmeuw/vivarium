# ~/ceam/tests/test_modules/test_module_registry.py

from unittest import TestCase

from ceam.modules import ModuleRegistry, DependencyException
from ceam.engine import SimulationModule


class BaseModule(SimulationModule):
    pass
class AModule(SimulationModule):
    pass
class BModule(SimulationModule):
    DEPENDENCIES = (AModule,)
class CModule(SimulationModule):
    DEPENDENCIES = (BModule,)
class DModule(SimulationModule):
    DEPENDENCIES = (CModule, BModule)


class TestModuleRegistration(TestCase):
    def test_register(self):
        registry = ModuleRegistry()

        # Registered modules are actually added to the internal store
        registry.register_modules([AModule()])
        self.assertSetEqual({m.__class__ for m in registry.modules}, {AModule})

        registry.register_modules([BModule()])
        self.assertSetEqual({m.__class__ for m in registry.modules}, {AModule, BModule})

        # Reregistering the same module is idempotent
        registry.register_modules([BModule()])
        self.assertEqual(len(registry.modules), 2)

        # Registering a module without it's dependencies implicitly registers those dependencies
        registry.register_modules([DModule()])
        self.assertSetEqual({m.__class__ for m in registry.modules}, {AModule, BModule, CModule, DModule})

    def test_sort_without_base_module(self):
        # TODO: this test is not complete. There are situations in practice where the sort is wrong but this passes.
        registry = ModuleRegistry()
        module_a = AModule()
        module_b = BModule()
        module_c = CModule()
        module_d = DModule()

        registry.register_modules([module_a, module_b])


#class TestSortModules(TestCase):
#    # TODO: this test is not complete. There are situations in practice where the sort is wrong but this passes.
#    def test_basic_function(self):
#        modules = {DModule: DModule(), CModule: CModule(), BModule:BModule(), AModule:AModule(), TestBaseModule:TestBaseModule()}
#        sorted_modules = sort_modules(modules, TestBaseModule)
#        self.assertListEqual(sorted_modules, [modules[TestBaseModule], modules[AModule], modules[BModule], modules[CModule], modules[DModule]])


# End.
