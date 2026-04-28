"""
===============
Event Interface
===============

The :class:`EventInterface` is exposed off the :ref:`builder <builder_concept>`
and provides two methods: :func:`get_emitter <EventInterface.get_emitter>`,
which returns a callable emitter for the given event type and
:func:`register_listener <EventInterface.register_listener>`, which adds the
given listener to the event channel for the given event. This is the only part
of the event framework with which client code should interact.

For more information, see the associated event :ref:`concept note <event_concept>`.

"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from vivarium.framework.event.manager import Event, EventManager
from vivarium.manager import Interface


class EventInterface(Interface):
    """The public interface for the :class:`~ <vivarium.framework.event.manager.Event>` system."""

    def __init__(self, manager: EventManager):
        self._manager = manager

    def get_emitter(
        self, event_name: str
    ) -> Callable[[pd.Index[int], dict[str, Any] | None], None]:
        """Gets an emitter for a named ``Event``.

        Parameters
        ----------
        event_name
            The name of the ``Event`` the requested emitter will emit.
            Users may provide their own named ``Events`` by requesting an emitter
            with this function, but should do so with caution as it makes time
            much more difficult to think about.

        Returns
        -------
            An emitter for the named ``Event``. The emitter should be called by
            the requesting component at the appropriate point in the simulation
            lifecycle.
        """
        return self._manager.get_emitter(event_name)

    def register_listener(
        self, event_name: str, listener: Callable[[Event], None], priority: int = 5
    ) -> None:
        """Registers a callable as a listener to an ``Event`` with the given name.

        The listening callable will be called with a named ``Event`` as its
        only argument any time the ``Event`` emitter is invoked from somewhere in
        the simulation.

        The framework creates the following ``Events`` and emits them at different
        points in the simulation:

            - At the end of the setup phase: ``post_setup``
            - Every time step:
              - ``time_step__prepare``
              - ``time_step``
              - ``time_step__cleanup``
              - ``collect_metrics``
            - At simulation end: ``simulation_end``

        Parameters
        ----------
        event_name
            The name of the ``Event`` to listen for.
        listener
            The callable to be invoked any time an ``Event`` with the given
            name is emitted.
        priority
            One of {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}.
            An indication of the order in which ``Event`` listeners should be
            called. Listeners with smaller priority values will be called
            earlier. Listeners with the same priority have no guaranteed
            ordering.  This feature should be avoided if possible. Components
            should strive to obey the Markov property as they transform the
            state table (the state of the simulation at the beginning of the
            next time step should only depend on the current state of the
            system).
        """
        self._manager.register_listener(event_name, listener, priority)
