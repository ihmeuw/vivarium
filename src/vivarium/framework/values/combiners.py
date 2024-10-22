from collections.abc import Callable
from typing import Any, Protocol


class ValueCombiner(Protocol):
    def __call__(
        self, value: Any, mutator: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        ...


def replace_combiner(
    value: Any, mutator: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Replace the previous pipeline output with the output of the mutator.

    This is the default combiner.

    Parameters
    ----------
    value
        The value from the previous step in the pipeline.
    mutator
        A callable that takes in all arguments that the pipeline source takes
        in plus an additional last positional argument for the value from
        the previous stage in the pipeline.
    args, kwargs
        The same args and kwargs provided during the invocation of the
        pipeline.

    Returns
    -------
        A modified version of the input value.
    """
    expanded_args = list(args) + [value]
    return mutator(*expanded_args, **kwargs)


def list_combiner(
    value: list[Any], mutator: Callable[..., Any], *args: Any, **kwargs: Any
) -> list[Any]:
    """Aggregates source and mutator output into a list.

    This combiner is meant to be used with a post-processor that does some
    kind of reduce operation like summing all values in the list.

    Parameters
    ----------
    value
        A list of all values provided by the source and prior mutators in the
        pipeline.
    mutator
        A callable that returns some portion of this pipeline's final value.
    args, kwargs
        The same args and kwargs provided during the invocation of the
        pipeline.

    Returns
    -------
        The input list with new mutator portion of the pipeline value
        appended to it.
    """
    value.append(mutator(*args, **kwargs))
    return value
