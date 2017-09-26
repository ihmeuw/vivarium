Dynamic Values Framework
========================
.. automodule:: ceam.framework.values

Decorators
----------
.. autofunction:: ceam.framework.values.produces_value
.. autofunction:: ceam.framework.values.modifies_value

Exceptions
----------
.. autoexception:: ceam.framework.values.DynamicValueError

Manager
-------
.. autoclass:: ceam.framework.values.ValuesManager
    :members:

Combiners and Post-processors
-----------------------------

These functions are used when constructing dynamic values with particular behaviors
and should rarely if ever be needed in client code.

.. autofunction:: ceam.framework.values.replace_combiner
.. autofunction:: ceam.framework.values.set_combiner
.. autofunction:: ceam.framework.values.joint_value_combiner

.. autofunction:: ceam.framework.values.rescale_post_processor
.. autofunction:: ceam.framework.values.joint_value_post_processor
