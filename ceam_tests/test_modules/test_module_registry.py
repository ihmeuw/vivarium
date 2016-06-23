# ~/ceam/tests/test_modules/test_module_registry.py

# To run ALL tests:  py.test (run from "~/ceam" directory).
# To run just tests in THIS file:  py.test tests/test_modules/test_module_registry.py  (run from same directory).


from unittest import TestCase

from ceam.modules import ModuleRegistry, DependencyException
from ceam.engine import SimulationModule


class BaseModule(SimulationModule):
    pass

class AModule(SimulationModule):
    pass

class BModule(SimulationModule):
    pass

class CModule(SimulationModule):
    pass

class DModule(SimulationModule):
    pass

class EModule(SimulationModule):
    DEPENDENCIES = (AModule, BModule)

class FModule(SimulationModule):
    DEPENDENCIES = (CModule, DModule)

class GModule(SimulationModule):
    DEPENDENCIES = (EModule, FModule)


class TestModuleRegistration(TestCase):
    def test_register(self):
        registry = ModuleRegistry()
        #
        # Registered modules are actually added to the internal store.
        registry.register_modules([AModule()])
        self.assertSetEqual({m.__class__ for m in registry.modules}, {AModule})
        #
        registry.register_modules([BModule()])
        self.assertSetEqual({m.__class__ for m in registry.modules}, {AModule, BModule})
        #
        # Reregistering the same module is idempotent.
        registry.register_modules([BModule()])
        self.assertEqual(len(registry.modules), 2)
        #
        # Registering a module without it's dependencies implicitly registers those dependencies.
        registry.register_modules([GModule()])
        self.assertSetEqual({m.__class__ for m in registry.modules}, {AModule, BModule, CModule, DModule, EModule, FModule, GModule})

    def insert_modules_in_order(self, output):
        resultset = set()
        for module in output:
            self.assertNotIn(module, resultset)
            for dependency in module.DEPENDENCIES:
                self.assertIn(dependency, resultset)
            resultset.add(module.__class__)

    def test_small_sort_without_base_module(self):
        registry = ModuleRegistry()
        module_a = AModule()
        module_b = BModule()
        module_e = EModule()
        #
        registry.register_modules([module_a, module_b, module_e])
        output = registry._ordered_modules
        #
        self.insert_modules_in_order(output)

    # Includes 3 levels (lacking BaseModule).
    def test_big_sort_without_base_module(self):
        registry = ModuleRegistry()
        module_a = AModule()
        module_b = BModule()
        module_c = CModule()
        module_d = DModule()
        module_e = EModule()
        module_f = FModule()
        module_g = GModule()
        #
        # Note that modules C, D, and F are not registered EXPLICITLY but should be registered IMPLICITLY because they are dependencies of G.
        registry.register_modules([module_a, module_b, module_e, module_g])
        output = registry._ordered_modules
        #
        self.insert_modules_in_order(output)

    # Includes 4 levels (including BaseModule).
    def test_big_sort_with_base_module(self):
        registry = ModuleRegistry(BaseModule)
        module_a = AModule()
        module_b = BModule()
        module_c = CModule()
        module_d = DModule()
        module_e = EModule()
        module_f = FModule()
        module_g = GModule()
        #
        # Note that modules C, D, and F are not registered EXPLICITLY but should be registered IMPLICITLY because they are dependencies of G.
        registry.register_modules([module_a, module_b, module_e, module_g])
        output = registry._ordered_modules
        #
        self.insert_modules_in_order(output)

    # This is a test to verify that the tests above are working correctly (ie, that the specification being tested is correct).  Rather than testing
    # a list sorted by the function-under-test (local function "inner_sort" in local function "_sort_modules" in class "ModuleRegistry"), it tests
    # a hand-sorted list of modules.  If the hand-sort succeeds while the software-sorted one fails, that points to a bug in the program.
    # Includes 4 levels (including BaseModule).
    def test_manual_sort_with_base_module(self):
        registry = ModuleRegistry(BaseModule)
        module_a = AModule()
        module_b = BModule()
        module_c = CModule()
        module_d = DModule()
        module_e = EModule()
        module_f = FModule()
        module_g = GModule()
        #
        # Note that all modules not EXPLICITLY registered should be registered IMPLICITLY because they are dependencies of G.
        registry.register_modules([module_g])
        manually_sorted_list = [BaseModule(), module_a, module_b, module_e, module_c, module_d, module_f, module_g]
        #
        self.insert_modules_in_order(manually_sorted_list)


# End.
