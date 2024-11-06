"""
=====================
The Logging Subsystem
=====================

"""

from __future__ import annotations

import loguru
from loguru import logger

from vivarium.framework.logging.utilities import configure_logging_to_terminal
from vivarium.manager import Interface, Manager


class LoggingManager(Manager):
    def __init__(self) -> None:
        self._simulation_name: str | None = None

    def configure_logging(
        self,
        simulation_name: str,
        verbosity: int = 0,
        long_format: bool = True,
    ) -> None:
        self._simulation_name = simulation_name
        if self._terminal_logging_not_configured():
            configure_logging_to_terminal(verbosity=verbosity, long_format=long_format)

    @staticmethod
    def _terminal_logging_not_configured() -> bool:
        # This hacks into the internals of loguru to see if we've already configured a
        # terminal sink. Loguru maintains a global increment of the loggers it has generated
        # and has a default logger configured with id 0. All code paths in this library that
        # configure logging handlers delete the default handler with id 0, add a terminal
        # logging handler (with id 1) and potentially have a file logging handler with id 2.
        # This behavior is based on sequencing of the handle definition. This is a bit
        # fragile since it depends on a loguru's internals as well as the stability of code
        # paths in vivarium, but both are quite stable at this point, so I think it's pretty,
        # low risk.
        return 1 not in logger._core.handlers  # type: ignore[attr-defined]

    @property
    def name(self) -> str:
        return "logging_manager"

    def get_logger(self, component_name: str | None = None) -> loguru.Logger:
        bind_args = {"simulation": self._simulation_name}
        if component_name:
            bind_args["component"] = component_name
        return logger.bind(**bind_args)


class LoggingInterface(Interface):
    def __init__(self, manager: LoggingManager) -> None:
        self._manager = manager

    def get_logger(self, component_name: str | None = None) -> loguru.Logger:
        return self._manager.get_logger(component_name)
