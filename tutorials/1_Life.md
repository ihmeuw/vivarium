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
from ceam.framework.population import uses_columns
from ceam.framework.event import listens_for
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

Here's the beginning of our population generation function:

```python
@uses_columns(['age', 'sex', 'alive'])
@listens_for('generate_population', priority=0)
def make_population(event):
```
The first decorator we use is `uses_columns` which tells the simulation which columns of the population store our function will use, modify or, in our case, create. `uses_columns` (and a couple of closely related methods we won't be using here) is the only way to modify the population in a CEAM simulation.

The second decorator is `listens_for` which tells the simulation that our function should be called when a 'generate_population' event happens. The `priority=0` says that we would like our function to be called before other functions that also listen for 'generate_population'.

The event system is a very important part of `CEAM`. Everything that happens in the simulation is driven by events and most of the functions you write will be called by `CEAM` in response to events that your code `listens_for`. The main event in the simulation is 'time_step' which happens every time the simulation moves the clock forward (in 30.5 day increments by default). Other events, like 'generate_population' happen before simulation time begins passing in order to give components a chance to do any preparation they need. You can get a list of all the events in the core `CEAM` system by running the command:
```sh
simulate list_events
```
