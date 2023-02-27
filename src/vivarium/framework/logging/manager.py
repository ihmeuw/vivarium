"""
=====================
The Logging Susbystem
=====================

"""
from loguru import logger
from loguru._logger import Logger

from vivarium.framework.logging.utilities import configure_logging_to_terminal


class LoggingManager:
    def __init__(self):
        self._simulation_name = None

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
        # terminal sink. This is a bit fragile since it depends on a libraries internals,
        # but loguru is a very stable library at this point, so I think it's pretty low risk.
        return 1 not in logger._core.handlers

    @property
    def name(self):
        return "logging_manager"

    def get_logger(self, component_name: str = None) -> Logger:
        if component_name:
            return logger.bind(simulation=self._simulation_name, component=component_name)
        return logger.bind(simulation=self._simulation_name)


class LoggingInterface:
    def __init__(self, manager: LoggingManager):
        self._manager = manager

    def get_logger(self, component_name: str = None) -> Logger:
        return self._manager.get_logger(component_name)
