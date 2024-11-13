from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any, Protocol

import pandas as pd

from vivarium.framework.utilities import from_yearly
from vivarium.framework.values.exceptions import DynamicValueError
from vivarium.types import NumberLike

if TYPE_CHECKING:
    from vivarium.framework.values.manager import ValuesManager


class PostProcessor(Protocol):
    def __call__(self, value: Any, manager: ValuesManager) -> Any:
        ...


def rescale_post_processor(value: NumberLike, manager: ValuesManager) -> NumberLike:
    """Rescales annual rates to time-step appropriate rates.

    This should only be used with a simulation using a
    :class:`~vivarium.framework.time.DateTimeClock` or another implementation
    of a clock that traffics in pandas date-time objects.

    Parameters
    ----------
    value
        Annual rates, either as a number or something we can broadcast
        multiplication over like a :mod:`numpy` array or :mod:`pandas`
        data frame.
    manager
        The ValuesManager for this simulation.

    Returns
    -------
        The annual rates rescaled to the size of the current time step size.
    """
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return value.mul(
            manager.simulant_step_sizes(value.index)
            .astype("timedelta64[ns]")
            .dt.total_seconds()
            / (60 * 60 * 24 * 365.0),
            axis=0,
        )
    else:
        time_step = manager.step_size()
        if not isinstance(time_step, (pd.Timedelta, timedelta)):
            raise DynamicValueError(
                "The rescale post processor requires a time step size that is a "
                "datetime timedelta or pandas Timedelta object."
            )
        return from_yearly(value, time_step)


def union_post_processor(values: list[NumberLike], _: Any) -> NumberLike:
    """Computes a probability on the union of the sample spaces in the values.

    Given a list of values where each value is a probability of an independent
    event, this post processor computes the probability of the union of the
    events.

    .. list-table::
       :width: 100%
       :widths: 1 3

       * - :math:`p_x`
         - Probability of event x
       * - :math:`1 - p_x`
         - Probability of not event x
       * - :math:`\prod_x(1 - p_x)`
         - Probability of not any events x
       * - :math:`1 - \prod_x(1 - p_x)`
         - Probability of any event x

    Parameters
    ----------
    values
        A list of independent proportions or probabilities, either
        as numbers or as a something we can broadcast addition and
        multiplication over.

    Returns
    -------
        The probability over the union of the sample spaces represented
        by the original probabilities.
    """
    # if there is only one value, return the value
    if len(values) == 1:
        return values[0]

    # if there are multiple values, calculate the joint value
    product: NumberLike = 1
    for v in values:
        new_value = 1 - v
        product = product * new_value
    joint_value = 1 - product
    return joint_value
