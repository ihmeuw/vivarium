"""
=================
Logging Interface
=================

This module provides an :class:`LoggingInterface <LoggingInterface>` class which
includes a method to get the logger system.

"""


from __future__ import annotations

import loguru

from vivarium.framework.logging.manager import LoggingManager
from vivarium.manager import Interface


class LoggingInterface(Interface):
    def __init__(self, manager: LoggingManager) -> None:
        self._manager = manager

    def get_logger(self, component_name: str | None = None) -> loguru.Logger:
        return self._manager.get_logger(component_name)
