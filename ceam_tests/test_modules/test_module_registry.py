# ~/ceam/tests/test_modules/test_module_registry.py

# To run ALL tests:  py.test (from "~/ceam" directory).
# To run just tests in THIS file:  py.test tests/test_modules/test_module_registry.py  (from same directory).


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

        # Registered modules are actually added to the internal store.
        registry.register_modules([AModule()])
        self.assertSetEqual({m.__class__ for m in registry.modules}, {AModule})

        registry.register_modules([BModule()])
        self.assertSetEqual({m.__class__ for m in registry.modules}, {AModule, BModule})

        # Reregistering the same module is idempotent.
        registry.register_modules([BModule()])
        self.assertEqual(len(registry.modules), 2)

        # Registering a module without it's dependencies implicitly registers those dependencies.
        registry.register_modules([DModule()])
        self.assertSetEqual({m.__class__ for m in registry.modules}, {AModule, BModule, CModule, DModule})

    def test_sort_1_without_base_module(self):
        registry = ModuleRegistry()
        module_a = AModule()
        module_b = BModule()

        registry.register_modules([module_a, module_b])
        output = registry._ordered_modules

        self.assertListEqual(output, [module_a, module_b])

    def test_sort_2_without_base_module(self):
        registry = ModuleRegistry()
        module_a = AModule()
        module_b = BModule()
        module_c = CModule()
        module_d = DModule()

        registry.register_modules([module_a, module_b, module_c, module_d])
        output = registry._ordered_modules

        self.assertListEqual(output, [module_a, module_b, module_c, module_d])


# End.
