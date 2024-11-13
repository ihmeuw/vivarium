"""
=========================
The Value Pipeline System
=========================

The value pipeline system is a vital part of the :mod:`vivarium`
infrastructure. It allows for values that determine the behavior of individual
:term:`simulants <Simulant>` to be constructed across multiple
:ref:`components <components_concept>`.

For more information about when and how you should use pipelines in your
simulations, see the value system :ref:`concept note <values_concept>`.

"""
from vivarium.framework.values.combiners import ValueCombiner, list_combiner, replace_combiner
from vivarium.framework.values.exceptions import DynamicValueError
from vivarium.framework.values.manager import ValuesInterface, ValuesManager
from vivarium.framework.values.pipeline import Pipeline, ValueModifier, ValueSource
from vivarium.framework.values.post_processors import (
    PostProcessor,
    rescale_post_processor,
    union_post_processor,
)
