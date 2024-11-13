"""
=================
Logging Utilities
=================

This module contains utilities for configuring logging.

"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from loguru import logger


def configure_logging_to_terminal(verbosity: int, long_format: bool = True) -> None:
    """Configure logging to print to the sys.stdout.

    Parameters
    ----------
    verbosity
        The verbosity level of the logging. 0 logs at the WARNING level, 1 logs
        at the INFO level, and 2 logs at the DEBUG level.
    long_format
        Whether to use the long format for logging messages, which includes explicit
        information about the simulation context and component in the log messages.
    """
    _clear_default_configuration()
    _add_logging_sink(
        sink=sys.stdout,
        verbosity=verbosity,
        long_format=long_format,
        colorize=True,
        serialize=False,
    )


def configure_logging_to_file(output_directory: Path) -> None:
    """Configure logging to write to a file in the provided output directory.

    Parameters
    ----------
    output_directory
        The directory to write the log file to.
    """
    log_file = output_directory / "simulation.log"
    _add_logging_sink(
        log_file,
        verbosity=2,
        long_format=True,
        colorize=False,
        serialize=False,
    )


def _clear_default_configuration() -> None:
    try:
        logger.remove(0)  # Clear default configuration
    except ValueError:
        pass


def _add_logging_sink(
    sink: Path | TextIO,
    verbosity: int,
    long_format: bool,
    colorize: bool,
    serialize: bool,
) -> int:
    """Add a logging sink to the logger.

    Parameters
    ----------
    sink
        The sink to add.  Can be a file path, a file object, or a callable.
    verbosity
        The verbosity level.  0 is the default and will only log warnings and errors.
        1 will log info messages.  2 will log debug messages.
    long_format
        Whether to use the long format for logging messages.  The long format includes
        the simulation name and component name.  The short format only includes the
        file name and line number.
    colorize
        Whether to colorize the log messages.
    serialize
        Whether to serialize log messages.  This is useful when logging to
        a file or a database.
    """
    log_formatter = _LogFormatter(long_format)
    logging_level = _get_log_level(verbosity)
    return logger.add(
        sink,
        colorize=colorize,
        level=logging_level,
        format=log_formatter.format,
        serialize=serialize,
    )


class _LogFormatter:
    time = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>"
    level = "<level>{level: <8}</level>"
    simulation = "<cyan>{extra[simulation]}</cyan> - <cyan>{name}</cyan>:<cyan>{line}</cyan>"
    simulation_and_component = (
        "<cyan>{extra[simulation]}</cyan>-<cyan>{extra[component]}</cyan>:<cyan>{line}</cyan>"
    )
    short_name_and_line = "<cyan>{name}</cyan>:<cyan>{line}</cyan>"
    message = "<level>{message}</level>"

    def __init__(self, long_format: bool = False):
        self.long_format = long_format

    if TYPE_CHECKING:
        from loguru import Record

    def format(self, record: Record) -> str:
        fmt = self.time + " | "

        if self.long_format:
            fmt += self.level + " | "

        if self.long_format and "simulation" in record["extra"]:
            if "component" in record["extra"]:
                fmt += self.simulation_and_component + " - "
            else:
                fmt += self.simulation + " - "
        else:
            fmt += self.short_name_and_line + " - "

        fmt += self.message + "\n{exception}"
        return fmt


def _get_log_level(verbosity: int) -> str:
    if verbosity == 0:
        return "WARNING"
    elif verbosity == 1:
        return "INFO"
    elif verbosity >= 2:
        return "DEBUG"
    else:
        raise ValueError(f"Invalid verbosity level: {verbosity}")


def list_loggers() -> None:
    """Utility function for analyzing the logging environment."""
    root_logger = logging.getLogger()
    print("Root logger: ", root_logger)
    for h in root_logger.handlers:
        print(f"     %s" % h)

    print("Other loggers")
    print("=============")
    for name, logger_ in logging.Logger.manager.loggerDict.items():
        print("+ [%-20s] %s " % (name, logger_))
        if not isinstance(logger_, logging.PlaceHolder):
            handlers = list(logger_.handlers)
            if not handlers:
                print("     No handlers")
            for h in logger_.handlers:
                print("     %s" % h)
