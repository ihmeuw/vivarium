Tutorial Two: Death
===================

In this tutorial we're going to make something happen to the simulants we created last time. We're going to make them die. We'll define a mortality rate and setup a new event listener than uses that rate to decide which simulants die during each time step.

Dynamic Rates
--------------
The next important `CEAM` concept we'll use is the dynamically calculated rates. Simulations are composed of many components which interact but may change from experiment to experiment so it's important that they be designed in such a way that they don't assume the presence of other components that are not absolutely necessary. Here's a concrete example. Say you're writing a simulation that models the effect of smoking on mortality. It's relatively simple to come up with a formula for that and include it in your mortality calculation. But what happens if in a future experiment you want to model BMI instead? Or both? To make those changes you'll need to modify the mortality calculation directly. `CEAM`'s dynamically calculated rates give us a way to model mortality (and similar rates) in such a way that models for things that effect mortality, like smoking and BMI, can be plugged in or unplugged transparently.

Dynamic rates in `CEAM` have globally unique names. So, mortality rate might be named 'mortality_rate' and any code that refers to that name is assumed to be referring to the same rate. Keep that in mind when naming rates. If you have an incidence rate, calling it 'incidence_rate' is likely to cause confusion since other components will have their own incidence rates. Better to use a more specific form like 'incidence_rate.heart_disease'. Each named rate has three important parts. The first is it's source which is a function that can calculate the base rate. For example 'mortality_rate' would need a source function that could calculate cause deleted mortality rates for each simulant. The second part are the mutators. These are any function that will change the rate from it's base form. An example would be a model of smoking that increases the mortality rate for simulants who are smokers. Components can add new mutators to a rate when they are added to the simulation. The third part is the rate consumers. These are any functions that use the final rate after any mutators have been applied to the base form. An example of a consumer is the code that uses the mortality rate to decide which simulants die. We'll create a named rate with a source and a consumer in this tutorial. The next tutorial will add a mutator.

### A note about rates and other types of dynamic values
> `CEAM` assumes that dynamic rates are yearly per capita rates unless specified otherwise. Rates are automatically scaled to the current time step size before they are returned to the consumer but sources and mutators should use raw yearly rate. `CEAM`'s dynamic value system is very flexible and other types of values, like joint PAF (population attributable fraction) values for risks, are possible and will be discussed later.

A Mortality Component
=====================

Like in the last tutorial, we'll present the full code for the component and then talk about it in detail.

```python
import pandas as pd
import numpy as np

from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam.framework.values import produces_value

class Mortality:
    def setup(self, builder):
        self.mortality_rate = builder.rate('mortality_rate')

    @produces_value('mortality_rate')
    def base_mortality_rate(self, index):
        return pd.Series(0.01, index=index)

    @listens_for('time_step')
    @uses_columns(['alive'], 'alive == True')
    def handler(self, event):
        effective_rate = self.mortality_rate(event.index)
        effective_probability = 1-np.exp(-effective_rate)
        draw = np.random.random(size=len(event.index))
        affected_simulants = draw < effective_probability
        event.population_view.update(pd.Series(False, index=event.index[affected_simulants]))

    @listens_for('simulation_end')
    @uses_columns(['alive'])
    def report(self, event):
       alive = event.population.alive.sum()
       dead = len(event.population) - alive
       print(alive, dead)
```

The imports are all the same as in the last tutorial except for one new one. We import `produces_value` which is a decorator that tells the simulation to use a function as the source for a named value, 'mortality_rate' in our case.

The major change is that this new component isn't a single function, it's a class with several methods and a bit of state. When `CEAM` is told to load a class as a component it will automatically instantiate it. If the class has a method called `setup` that will be called during simulation startup and can be used to initialize internal state or acquire references to things in the simulation, like mutable rates, which the component will need.

Our setup looks like this:
```python
    def setup(self, builder):
        self.mortality_rate = builder.rate('mortality_rate')
```

The only thing we do here is we get a reference to the dynamic rate named 'mortality_rate' which we use later do determine the probability of a simulant dying. The `builder` object is a simple container for several functions that let us get these kinds of references out of the simulation during setup. We'll see more examples of what it makes available later.

Next we have the source function for our 'mortality_rate':

```python
    @produces_value('mortality_rate')
    def base_mortality_rate(self, index):
        return pd.Series(0.01, index=index)
```

This creates a `pandas.Series` which in analogous to a single column from a `DataFrame` where each simulant in the requested index is assigned a rate of 0.01 per year. A more realistic example would assign individual rates to each simulant based on some model of mortality and, in fact, we will refine this in future tutorials.

Next we get to the meat of the component, the function which decides which simulants die:
```python
    @listens_for('time_step')
    @uses_columns(['alive'], 'alive == True')
    def handler(self, event):
        effective_rate = self.mortality_rate(event.index)
        effective_probability = 1-np.exp(-effective_rate)
        draw = np.random.random(size=len(event.index))
        affected_simulants = draw < effective_probability
        event.population_view.update(pd.Series(False, index=event.index[affected_simulants]), name='alive')
```

 It is similar to the 'generate_population' listener we created in the last tutorial except that it listens to 'time_step' instead which means that, rather than being called one time before the simulation starts, it will be called for every tick of the simulation's clock which happen at 30.5 day intervals. It also uses only a single column from the population table, 'alive'. There is also a new, second argument to the `uses_columns` decorator which causes the simulation to filter the population before passing it to our function. In this case we only want to see simulants who susceptible, ie. still alive. No point in killing them twice.

Next we need to get the effective mortality rate for each simulant in the susceptible population. We do that by calling the `mortality_rate` function that we got out of the builder during the setup phase. This will cause the simulation to query the rate's source, in this case the `base_mortality_rate` method on our class though it could just as easily be located in some other component. The value would then be passed through the value's mutators if there were any. Finally the rates are rescaled from yearly to monthly to match the size of our time step and returned.

We then convert the rate into a probability and get a random number between 0 and 1 for each simulant. We compare the random numbers to the probabilities using the standard less than sign. When comparing `pandas` and `numpy` data structures like `Series` the result will be a list of `True` or `False` values one for each row representing the truth of the comparison at that row. So, `affected_simulants` will be a list with `True` for each row where the random number was less than the probability, meaning that simulant has died, and `False` otherwise.

We then use `affected_simulants` as a filter on the index so we only have the subset of the index corresponding to those simulants who are now dead which we use to construct a `pandas.Series` of `False`s and use that to update the underlying population table.

At this point we have enough to run the simulation and have the mortality rate effect the population but we still can't see what's happening. The last function in the class adds some reporting to show us how many people died:
```python
    @listens_for('simulation_end')
    @uses_columns(['alive'])
    def report(self, event):
       alive = event.population.alive.sum()
       dead = len(event.population) - alive
       print('Living simulants: {} Dead simulants: {}'.format(alive, dead))
```

This uses a new event 'simulation_end' which, as you might expect, happens at the end of the simulation run. It also uses `event.population` which, like `event.index`, contains the simulant's effected by the event (in this case all of them) but rather than just having their position in the population table as the index does it has all the data we requested through `uses_columns`, so just the 'alive' column. We use the `sum()` aggregation on that column (in python `True` evaluates to 1 and `False` evaluates to 0) to count the living simulants. Invert that and you have a count of the dead which will get printed out after the last time step.

Update the configuration.json file to include the new component:
```json
{
    "components": [
        "ceam_tutorial.components.initial_population.make_population",
        "ceam_tutorial.components.mortality.Mortality"
    ]
}
```

And run:
```sh
> simulate configuration.json
```

In the [next tutorial](./3_The_Part_In_Between.md) we'll make this all a bit more complex and a bit more realistic.

Another Exercise For The Reader
--------------------------------

Now that you've seen an implementation of mortality you should be able to build out a simple model of a disease where people start healthy and get sick at a fixed rate just like our fixed mortality rate. Look back at the last tutorial where we created the 'age', 'sex' and 'alive' columns. You can add a listener for the 'generate_population' event to your disease component that creates a column to record whether or not people are sick, just like the 'alive' column records whether or not they are alive.

And An Extension To The Last One
--------------------------------

Remember in the last exercise where I asked you how you would make the heights you assigned more realistic? One way to do it would be by making children shorter than adults but you probably didn't know how old people were in your height component. Now you've seen how we find out who is alive in the `report` function. Can you do the same thing with age in your component?
