Dynamic Values Framework
========================
.. automodule:: vivarium.framework.values

Decorators
----------
.. autofunction:: vivarium.framework.values.produces_value
.. autofunction:: vivarium.framework.values.modifies_value

Exceptions
----------
.. autoexception:: vivarium.framework.values.DynamicValueError

Manager
-------
.. autoclass:: vivarium.framework.values.ValuesManager
    :members:

Combiners and Post-processors
-----------------------------

These functions are used when constructing dynamic values with particular behaviors
and should rarely if ever be needed in client code.

.. autofunction:: vivarium.framework.values.replace_combiner
.. autofunction:: vivarium.framework.values.set_combiner
.. autofunction:: vivarium.framework.values.joint_value_combiner

.. autofunction:: vivarium.framework.values.rescale_post_processor
.. autofunction:: vivarium.framework.values.joint_value_post_processor
