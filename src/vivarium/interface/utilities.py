import functools

from vivarium import VivariumError
from vivarium.framework.configuration import validate_model_specification_file


def verify_yaml(_, __, file_path: str) -> str:
    """Ensures the provided file exists and is a yaml file"""
    return validate_model_specification_file(file_path)


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
