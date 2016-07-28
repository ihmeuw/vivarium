# ~/ceam/ceam_tests/test_modules/test_module_registry.py

# The dependency tree associated with the "big" tests below follows.  Classes are "above" those upon which they depend.
# If BaseModule is included, then ALL classes depend upon it).
#
#                          GModule                         HModule                           IModule
#                        /         \                       /     \                          /       \
#                 EModule           FModule            BModule  CModule             _______/        FModule
#                /       \         /       \                                       /      /          /    \
#            AModule  BModule  CModule  DModule                               AModule  BModule  CModule  DModule


from unittest import TestCase

from ceam.tree import Node
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

class HModule(SimulationModule):
    DEPENDENCIES = (BModule, CModule)

class IModule(SimulationModule):
    DEPENDENCIES = (AModule, BModule, FModule)

class TestModuleRegistry(Node, ModuleRegistry):
    pass


class TestModuleRegistration(TestCase):
    def test_register(self):
        registry = TestModuleRegistry()
        #
        # Registered modules are actually added to the internal store.
        registry.add_children([AModule()])
        self.assertSetEqual({m.__class__ for m in registry.modules}, {AModule})
        #
        registry.add_children([BModule()])
        self.assertSetEqual({m.__class__ for m in registry.modules}, {AModule, BModule})
        #
        # Reregistering the same module is idempotent.
        registry.add_children([BModule()])
        self.assertEqual(len(registry.modules), 2)
        #
        # Registering a module without it's dependencies implicitly registers those dependencies.
        registry.add_children([GModule()])
        self.assertSetEqual({m.__class__ for m in registry.modules}, {AModule, BModule, CModule, DModule, EModule, FModule, GModule})

    def insert_modules_in_order(self, output):
        resultset = set()
        for module in output:
            self.assertNotIn(module, resultset)
            for dependency in module.DEPENDENCIES:
                self.assertIn(dependency, resultset)
            resultset.add(module.__class__)

    # Includes 2 levels (lacking BaseModule).
    def test_small_sort_without_base_module(self):
        registry = TestModuleRegistry()
        module_a = AModule()
        module_b = BModule()
        module_e = EModule()
        #
        registry.add_children([module_a, module_b, module_e])
        output = registry.modules
        #
        self.insert_modules_in_order(output)

    # Includes 2 levels (lacking BaseModule).
    def test_alt_small_sort_without_base_module(self):
        registry = TestModuleRegistry()
        module_b = BModule()
        module_c = CModule()
        module_h = HModule()
        #
        # Note that modules BModule and CModule are not registered EXPLICITLY but should be registered IMPLICITLY because they are dependencies of HModule.
        registry.add_children([module_h])
        output = registry.modules
        #
        self.insert_modules_in_order(output)

    # Includes 3 levels (lacking BaseModule).
    def test_big_sort_without_base_module(self):
        registry = TestModuleRegistry()
        module_a = AModule()
        module_b = BModule()
        module_c = CModule()
        module_d = DModule()
        module_e = EModule()
        module_f = FModule()
        module_g = GModule()
        #
        # Note that modules CModule, DModule, and FModule are not registered EXPLICITLY but should be registered IMPLICITLY because they are dependencies of GModule.
        registry.add_children([module_a, module_b, module_e, module_g])
        output = registry.modules
        #
        self.insert_modules_in_order(output)

    # Includes 4 levels (including BaseModule).
    def test_big_sort_with_base_module(self):
        registry = TestModuleRegistry(BaseModule)
        module_a = AModule()
        module_b = BModule()
        module_c = CModule()
        module_d = DModule()
        module_e = EModule()
        module_f = FModule()
        module_g = GModule()
        #
        # Note that modules CModule, DModule, and FModule are not registered EXPLICITLY but should be registered IMPLICITLY because they are dependencies of GModule.
        registry.add_children([module_a, module_b, module_e, module_g])
        output = registry.modules
        #
        self.insert_modules_in_order(output)

    # Includes 4 levels (including BaseModule).
    def test_alt_big_sort_with_base_module(self):
        registry = TestModuleRegistry(BaseModule)
        module_a = AModule()
        module_b = BModule()
        module_c = CModule()
        module_d = DModule()
        module_f = FModule()
        module_i = GModule()
        #
        # Note that most modules are not registered EXPLICITLY but should be registered IMPLICITLY because they are dependencies of IModule.
        registry.add_children([module_i])
        output = registry.modules
        #
        self.insert_modules_in_order(output)


# End.
