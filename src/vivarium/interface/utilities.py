"""
===========================
Interface Utility Functions
===========================

The functions defined here are used to support the interactive and command-line
interfaces for ``vivarium``.

"""
from datetime import datetime
import functools
from pathlib import Path
import sys

from loguru import logger

from vivarium.exceptions import VivariumError


def run_from_ipython() -> bool:
    """Taken from https://stackoverflow.com/questions/5376837/how-can-i-do-an-if-run-from-ipython-test-in-python"""
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def log_progress(sequence, every=None, size=None, name='Items'):
    """Taken from https://github.com/alexanderkuk/log-progress"""
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except Exception as e:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )


class InteractiveError(VivariumError):
    """Error raised when the Interactive context is in an inconsistent state."""
    pass


def raise_if_not_setup(system_type):
    type_error_map = {
        'run': 'Simulation must be setup before it can be run',
        'value': 'Value pipeline configuration is not complete until the simulation is setup.',
        'event': 'Event configuration is not complete until the simulation is setup.',
        'component': 'Component configuration is not complete until the simulation is setup.',
        'population': 'No population exists until the simulation is setup.',
    }
    err_msg = type_error_map[system_type]

    def method_wrapper(context_method):

        @functools.wraps(context_method)
        def wrapped_method(*args, **kwargs):
            instance = args[0]
            if not instance._setup:
                raise InteractiveError(err_msg)
            return context_method(*args, **kwargs)

        return wrapped_method

    return method_wrapper


def get_output_root(results_directory, model_specification_file):
    launch_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_name = Path(model_specification_file).stem
    output_root = Path(results_directory + f"/{model_name}/{launch_time}")
    return output_root


def add_logging_sink(sink, verbose, colorize=False, serialize=False):
    message_format = ('<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | '
                      '<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> '
                      '- <level>{message}</level>')
    if verbose:
        logger.add(sink, colorize=colorize, level="DEBUG", format=message_format, serialize=serialize)
    else:
        logger.add(sink, colorize=colorize, level="ERROR", format=message_format, serialize=serialize)


def configure_logging_to_terminal(verbose):
    logger.remove(0)  # Clear default configuration
    add_logging_sink(sys.stdout, verbose, colorize=True)


def configure_logging_to_file(output_directory):
    master_log = output_directory / 'simulation.log'
    add_logging_sink(master_log, verbose=True)
