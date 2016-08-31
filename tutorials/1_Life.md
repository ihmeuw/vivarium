Tutorial One: Life
===================

In this tutorial we're going to create an initial population for our simulation and in the process talk about some of the basic concepts in `CEAM`.

But First, Some Terminology
--------------------------

* __simulant__: An individual member of the population being simulated.
* __component__: Any self contained piece of code that can be plugged into the simulation to add some functionality. For example, code that modeled simulants' blood pressures would be a component.

Our First Component
-------------------

We're going to use the convention that every component is contained inside a it's own python file (a single module) so we'll need a file for our population generator. Let's call the file `initial_population.py` and put it in the `components` directory.

First we need to import some tools:

```python
import numpy as np
mport pandas as pd
```

`numpy` is a library for doing high performance numerical computing in python. `pandas` is a set of tools built on top of `numpy` that allow for fast data base style querying and aggregating of data. `CEAM` uses `pandas` `DataFrame` objects (similar to R's Data Frames) to store much of it's data so `pandas` is very important for using `CEAM`.

Next we import some decorators from `CEAM`:

```python
from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
```

### About Decorators

> Decorators are used to modify or label functions. They come immediately before a function and are prefixed with the @-sign.
```python
@my_decorator
def my_function():
    print('Doing something...')
```
> In this example the decorator `my_decorator` is applied to the function `my_function`. Decorators can be used to change a function's behavior, for example causing the function to print out timing information after it finishes running. They can also be used to mark a function as having some property, for example a decorator might be used to mark a test function as particularly slow so that a testing framework can choose to skip it or run it later when speed is a concern.
> `CEAM` uses decorators to tell the simulation how it is expected to use functions within a component and what data those functions will need to access.
