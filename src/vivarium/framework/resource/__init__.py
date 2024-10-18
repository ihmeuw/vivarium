"""
===================
Resource Management
===================

This module provides a tool to manage dependencies on resources within a
:mod:`vivarium` simulation. These resources take the form of things that can
be created and utilized by components, for example columns in the
:mod:`state table <vivarium.framework.population>`
or :mod:`named value pipelines <vivarium.framework.values>`.

Because these resources need to be created before they can be used, they are
sensitive to ordering. The intent behind this tool is to provide an interface
that allows other managers to register resources with the resource manager
and in turn ask for ordered sequences of these resources according to their
dependencies or raise exceptions if this is not possible.

For more information, see the Resource Management
:ref:`concept note<resource_concept>`.

"""

from vivarium.framework.resource.manager import ResourceInterface, ResourceManager
from vivarium.framework.resource.resource import Resource
