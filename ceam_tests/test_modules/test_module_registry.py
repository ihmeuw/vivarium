# ~/ceam/tests/test_modules/test_module_registry.py

# To run ALL tests:  py.test (run from "~/ceam" directory).
# To run just tests in THIS file:  py.test tests/test_modules/test_module_registry.py  (run from same directory).


# Explanation of the difference between "level-only" and "depth-first strictness" tests:
#
#   The "level-only" spec states merely that classes at a lower level in the hierarchy must come before those higher.  A "lower" class means something "above" it
#   depends upon it.  A "higher" class means it depends upon something "lower" than it.  But WHICH items depend upon WHICH OTHERS is ignored - only "level" matters.
#
#   The "depth-first" spec specifies WHICH classes depend upon WHICH OTHERS.  In other words, a class is listed IMMEDIATELY FOLLOWING a class (or group of classes)
#   that it depends upon (not merely somewhere later in the listing, as can happen in the "level-only" specification).
#
#   For example, the dependency tree associated with the "big" tests here follows (classes are "above" those upon which they depend; if BaseModule is included,
#   then ALL classes depend upon it):
#
#                        GG
#                      /    \
#                    EE      FF
#                   /  \    /  \
#                  AA  BB  CC  DD
#
#   The "level-only" requirement is that any from AA, BB, CC, DD must come (in any order) before EE or FF (in any order), and GG must be last.
#   Thus [AA, CC, DD, BB, FF, EE, GG] is a legal order.
#
#   The "depth-first" requirement is that any dependent element must immediately follow those it depends on.  Thus the order immediately above is NOT valid
#   (because EE does not immediately follow AA and BB), but [AA, BB, EE, DD, CC, FF, GG] IS valid.  This order is the result of a depth-first tree traversal.
#
#   For the sort algorithm used (and being tested here), the REQUIREMENT is the specification of "level-only".  However, the ALGORITHM used is a depth-first traversal.
#   Therefore, the "level-only" tests can be thought of as test of functional correctness AS SPECIFIED, while the "depth-first" tests are a measure of the correctness
#   of the algorithm currently used.  The "depth-first" tests are therefore NOT required to pass as "black-box unit tests" but will be useful as tools for debugging
#   the current algorithm (ie, as "glass-box" tests).


from unittest import TestCase

from ceam.modules import ModuleRegistry, DependencyException
from ceam.engine import SimulationModule


class BaseModule(SimulationModule):
    LEVEL = 0

class AModule(SimulationModule):
    LEVEL = 1

class BModule(SimulationModule):
    LEVEL = 1

class CModule(SimulationModule):
    LEVEL = 1

class DModule(SimulationModule):
    LEVEL = 1

class EModule(SimulationModule):
    LEVEL = 2
    DEPENDENCIES = (AModule, BModule)

class FModule(SimulationModule):
    LEVEL = 2
    DEPENDENCIES = (CModule, DModule)

class GModule(SimulationModule):
    LEVEL = 3
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

    # There is only a single test here since level-only and depth-first strictness are equivalent in this case.  Includes 2 levels only; no BaseModule.
    def test_small_sort_without_base_module(self):
        registry = ModuleRegistry()
        module_a = AModule()
        module_b = BModule()
        module_e = EModule()
        #
        registry.register_modules([module_a, module_b, module_e])
        output = registry._ordered_modules
        #
        self.assertListEqual(output, [module_a, module_b, module_e])

    # The level-only test here is a less strict specification for the sort than the depth-first strictness requirement.  Includes 3 levels (lacking BaseModule).
    def test_big_sort_level_only_without_base_module(self):
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
        levels = [cls.LEVEL for cls in output]
        #
        self.assertListEqual(levels, [1, 1, 1, 1, 2, 2, 3]

    # The depth-first strictness test here is a stricter specification for the sort than the level-only requirement.  Includes 3 levels (lacking BaseModule).
    def test_big_sort_depth_first_without_base_module(self):
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
        self.assertIn(output, [[module_a, module_b, module_e, module_c, module_d, module_f, module_g],
                               [module_b, module_a, module_e, module_c, module_d, module_f, module_g],
                               [module_a, module_b, module_e, module_d, module_c, module_f, module_g],
                               [module_b, module_a, module_e, module_d, module_c, module_f, module_g],
                               [module_c, module_d, module_f, module_a, module_b, module_e, module_g],
                               [module_c, module_d, module_f, module_b, module_a, module_e, module_g],
                               [module_d, module_c, module_f, module_a, module_b, module_e, module_g],
                               [module_d, module_c, module_f, module_b, module_a, module_e, module_g]])

    # The level-only test here is a less strict specification for the sort than the depth-first strictness requirement.  Includes 4 levels (including BaseModule).
    def test_big_sort_level_only_with_base_module(self):
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
        levels = [cls.LEVEL for cls in output]
        #
        self.assertListEqual(levels, [0, 1, 1, 1, 1, 2, 2, 3]
        self.assertIn(levels, [[BaseModule, module_a, module_b, module_e, module_c, module_d, module_f, module_g],
                               [BaseModule, module_b, module_a, module_e, module_c, module_d, module_f, module_g],
                               [BaseModule, module_a, module_b, module_e, module_d, module_c, module_f, module_g],
                               [BaseModule, module_b, module_a, module_e, module_d, module_c, module_f, module_g],
                               [BaseModule, module_c, module_d, module_f, module_a, module_b, module_e, module_g],
                               [BaseModule, module_c, module_d, module_f, module_b, module_a, module_e, module_g],
                               [BaseModule, module_d, module_c, module_f, module_a, module_b, module_e, module_g],
                               [BaseModule, module_d, module_c, module_f, module_b, module_a, module_e, module_g]])

    # The depth-first strictness test here is a stricter specification for the sort than the level-only requirement.  Includes 4 levels (including BaseModule).
    def test_big_sort_depth_first_with_base_module(self):
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
        self.assertIn(output, [[BaseModule, module_a, module_b, module_e, module_c, module_d, module_f, module_g],
                               [BaseModule, module_b, module_a, module_e, module_c, module_d, module_f, module_g],
                               [BaseModule, module_a, module_b, module_e, module_d, module_c, module_f, module_g],
                               [BaseModule, module_b, module_a, module_e, module_d, module_c, module_f, module_g],
                               [BaseModule, module_c, module_d, module_f, module_a, module_b, module_e, module_g],
                               [BaseModule, module_c, module_d, module_f, module_b, module_a, module_e, module_g],
                               [BaseModule, module_d, module_c, module_f, module_a, module_b, module_e, module_g],
                               [BaseModule, module_d, module_c, module_f, module_b, module_a, module_e, module_g]])


# End.
