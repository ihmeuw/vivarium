.. _results_concept:

==================
Results Management
==================

.. contents::
   :depth: 2
   :local:
   :backlinks: none

The results management system is responsible for collecting, formatting, and storing
results of a ``vivarium`` simulation. The results are then formatted and written
to disk at the end of the simulation during the :ref:`simulation end phase <lifecycle_simulation_end>`.

Main Concepts
-------------

There are three main concepts that make up the results management system:
observers, observations, and stratifications. An **observer** registers desired
measure-specific results - referred to as **observations** - that may or may not 
be grouped into **stratifications**.

.. note::
    A ``vivarium`` simulation will *not* record results by default. The user must
    define observers that register observations in order to record results!

.. note::
    Users should not interact with observations and stratifications directly - 
    they should only be created by the methods provided by the 
    :class:`ResultsInterface <vivarium.framework.results.interface.ResultsInterface>`.

How to Use the Results Management System
----------------------------------------

The intended workflow for using the results system is to create an **observer**
and register **observations** with it. **Stratifications** can also be registered 
if desired.

Creating an Observer and Registering Observations
++++++++++++++++++++++++++++++++++++++++++++++++++

All **observers** should be concrete instances of the 
:class:`Observer <vivarium.framework.results.observer.Observer>` 
abstract base class which guarantees that it is a proper ``vivarium`` 
:class:`Component <vivarium.component.Component>`. While the user is free to 
add whatever business logic is necessary, the primary goal of the component lies 
in the :meth:`register_observations <vivarium.framework.results.observer.Observer.register_observations>`
method. This is a required method (indeed, it is an abstract method of the 
**Observer** class) and is where the user should register all observations 
(ideally one per observer).

Observation registration methods exist on the simulation's 
:class:`ResultsInterface <vivarium.framework.results.interface.ResultsInterface>` and 
can be accessed through the :ref:`builder <builder_concept>`:

- :meth:`builder.results.register_stratified_observation <vivarium.framework.results.interface.ResultsInterface.register_stratified_observation>`
- :meth:`builder.results.register_unstratified_observation <vivarium.framework.results.interface.ResultsInterface.register_unstratified_observation>`
- :meth:`builder.results.register_adding_observation <vivarium.framework.results.interface.ResultsInterface.register_adding_observation>`
- :meth:`builder.results.register_concatenating_observation <vivarium.framework.results.interface.ResultsInterface.register_concatenating_observation>`

For example, here is an an observer that records the number of deaths in a simulation
(defined completely through the "pop_filter" argument). That is, it records the 
number of people that have died during the current time step and adds that number 
to the existing number of people who have died from previous time steps. 

.. testcode::

  from typing import Any, Optional

  import pandas as pd

  from vivarium.framework.engine import Builder
  from vivarium.framework.results import Observer

  class DeathObserver(Observer):

    @property
    def configuration_defaults(self) -> dict[str, Any]:
      return {
        "mortality": {
          "life_expectancy": 80,
        }
      }

    @property
    def columns_required(self) -> Optional[list[str]]:
      return ["age", "alive"]

    def register_observations(self, builder: Builder) -> None:
      builder.results.register_adding_observation(
        name="total_population_dead",
        requires_columns=["alive"],
        pop_filter='alive == "dead"',
      )

And here is an example of how you might create an observer that records new 
births (defined completely through the "pop_filter" argument), concatenates them 
to existing ones, and formats the data to only include specified state table columns 
as well as adds a new one ("birth_date").

.. testcode::

  from datetime import datetime

  import pandas as pd

  from vivarium.framework.engine import Builder
  from vivarium.framework.results import Observer

  class BirthObserver(Observer):

    COLUMNS = ["sex", "birth_weight", "gestational_age", "pregnancy_outcome"]

    def register_observations(self, builder: Builder) -> None:
      builder.results.register_concatenating_observation(
        name="births",
        pop_filter=(
          "("
          f"pregnancy_outcome == 'live_birth' "
          f"or pregnancy_outcome == 'stillbirth'"
          ") "
          f"and previous_pregnancy == 'pregnant' "
          f"and pregnancy == 'parturition'"
        ),
        requires_columns=self.COLUMNS,
        results_formatter=self.format,
      )

    def format(self, measure: str, results: pd.DataFrame) -> pd.DataFrame:
      new_births = results[self.COLUMNS]
      new_births["birth_date"] = datetime(2024, 12, 30).strftime("%Y-%m-%d T%H:%M.%f")
      return new_births

As both of these examples are proper ``vivarium``
:class:`Components <vivarium.component.Component>`, they are added to the simulation
via the :ref:`model specification <model_specification_concept>` like any other component:

.. code-block:: yaml

  components:
    <PACKAGE_NAME>:
      <SUBDIR>:  # as many subdirectories as needed to fully define the path
        - DeathObserver()
        - BirthObserver()

Stratifying Observations
++++++++++++++++++++++++

If you want to stratify the results of an **observation** (that is, group and 
aggregate by designated categories), you can register a 
:class:`Stratification <vivarium.framework.results.stratification.Stratification>` 
with the results system. Stratification registration methods exist on the simulation's 
:class:`ResultsInterface <vivarium.framework.results.interface.ResultsInterface>` and 
can be accessed through the :ref:`builder <builder_concept>`:

- :meth:`builder.results.register_stratification <vivarium.framework.results.interface.ResultsInterface.register_stratification>`
- :meth:`builder.results.register_binned_stratification <vivarium.framework.results.interface.ResultsInterface.register_binned_stratification>`

Here is an example of how you might register a "current_year" and "sex" as stratifications:

.. testcode::

  import pandas as pd

  from vivarium import Component
  from vivarium.framework.engine import Builder

  class ResultsStratifier(Component):
    """Register stratifications for the results system"""

    def setup(self, builder: Builder) -> None:
      self.start_year = builder.configuration.time.start.year
      self.end_year = builder.configuration.time.end.year
      self.register_stratifications(builder)

    def register_stratifications(self, builder: Builder) -> None:
      builder.results.register_stratification(
        "current_year",
        [str(year) for year in range(self.start_year, self.end_year + 1)],
        mapper=self.map_year,
        is_vectorized=True,
        requires_columns=["current_time"],
      )
      builder.results.register_stratification(
        "sex", ["Female", "Male"], requires_columns=["sex"]
      )

    ###########
    # Mappers #
    ###########

    @staticmethod
    def map_year(pop: pd.DataFrame) -> pd.Series:
      """Map datetime with year

      Parameters
      ----------
      pop
        A pd.DataFrame with one column, a datetime to be mapped to year

      Returns
      ------
      pandas.Series
        A pd.Series with years corresponding to the pop passed into the function
      """
      return pop.squeeze(axis=1).dt.year.apply(str)

.. note::
  Good encapsulation suggests that all stratification registrations occur in a single
  class (as in the **ResultsStratifier** class in the above example). This is not
  enforced, however, and it is also somewhat common to register a stratification 
  that will only be used by a single observer within that observer's 
  :meth:`register_observations <vivarium.framework.results.observer.Observer.register_observations>`
  method.

Just because you've *registered* a stratification doesn't mean that the results will
actually *use* it. In order to use the stratification, you must add it to the 
:ref:`model specification <model_specification_concept>` configuration block 
using the "stratification" key. You can provide "default" stratifications which 
will be used by all observations as well as observation-specific "include" and 
"exclude" keys to further modify each observation's stratifications.

For example, to use "age_group" and "sex" as default stratifications for *all* 
observations and then customize "deaths" observations to also include 
"location" but not "age_group":

.. code-block:: yaml

  configuration:
    stratification:
      default:
        - 'age_group'
        - 'sex'
      deaths:
          include: ['location']
          exclude: ['age_group']

.. note::
    All stratifications must be included as a list, even if there is only one.

Excluding Categories from Results
+++++++++++++++++++++++++++++++++

It is also possible to exclude specific *categories* from results processing. For 
example, perhaps we do not care about results for simulants in certain "age_groups" 
or who have a certain "cause_of_death" or "disability". Excluding categories 
is done by providing an "excluded_categories" key along with a *per observation* 
list of categories to exclude within the :ref:`model specification's <model_specification_concept>` 
stratification block. 

For example, to exclude "stillbirth" as a pregnancy outcome during results processing:

.. code-block:: yaml

  configuration:
    stratification:
      default:
        - 'age_group'
        - 'sex'
      births:
          include: ['pregnancy_outcome']
          exclude: ['age_group']
      excluded_categories:
        pregnancy_outcome: ['stillbirth']

Observers
---------

The :class:`Observer <vivarium.framework.results.observer.Observer>` object is a 
``vivarium`` :class:`Component <vivarium.component.Component>` and abstract base 
class whose primary purpose is to register observations to the results system. 
Ideally, each concrete observer class should register a single observation (though 
this is not enforced).

Observations
------------

When discussing the results system, an **observation** is used somewhat interchangeably
with the term "results". More specifically, an observation is a set of measure-specific
results that are collected throughout the simulation.

Implementation-wise, an observation is a data structure that holds the values 
and callables required to collect the results of a specific measure during the simulation. 

At the highest level, an observation can be considered either *stratified* or
*unstratified*. A 
:class:`StratifiedObservation <vivarium.framework.results.observation.StratifiedObservation>`
is one whose results are grouped into and aggregated by categories referred to as 
**stratifications**. An 
:class:`UnstratifiedObservation <vivarium.framework.results.observation.UnstratifiedObservation>`
is one whose results are not grouped into categories.

A couple other more specific and commonly used observations are provided as well:

- :class:`AddingObservation <vivarium.framework.results.observation.AddingObservation>`: 
  a specific type of 
  :class:`StratifiedObservation <vivarium.framework.results.observation.StratifiedObservation>` 
  that gathers new results and adds/sums them to any existing results.
- :class:`ConcatenatingObservation <vivarium.framework.results.observation.ConcatenatingObservation>`: 
  a specific type of 
  :class:`UnstratifiedObservation <vivarium.framework.results.observation.UnstratifiedObservation>` 
  that gathers new results and concatenates them to any existing results.

Ideally, all concrete classes should inherit from the 
:class:`BaseObservation <vivarium.framework.results.observation.BaseObservation>`
abstract base class, which contains the common attributes between observation types:

.. list-table:: **Common Observation Attributes**
  :widths: 15 45
  :header-rows: 1

  * - Attribute
    - Description
  * - | :attr:`name <vivarium.framework.results.observation.BaseObservation.name>`
    - | Name of the observation. It will also be the name of the output results file
      | for this particular observation.
  * - | :attr:`pop_filter <vivarium.framework.results.observation.BaseObservation.pop_filter>`
    - | A Pandas query filter string to filter the population down to the simulants
      | who should be considered for the observation.
  * - | :attr:`when <vivarium.framework.results.observation.BaseObservation.when>`
    - | Name of the lifecycle phase the observation should happen. Valid values are:
      | "time_step__prepare", "time_step", "time_step__cleanup", or "collect_metrics".
  * - | :attr:`results_initializer <vivarium.framework.results.observation.BaseObservation.results_initializer>`
    - | Method or function that initializes the raw observation results
      | prior to starting the simulation. This could return, for example, an empty
      | DataFrame or one with a complete set of stratifications as the index and
      | all values set to 0.0.
  * - | :attr:`results_gatherer <vivarium.framework.results.observation.BaseObservation.results_gatherer>`
    - | Method or function that gathers the new observation results.
  * - | :attr:`results_updater <vivarium.framework.results.observation.BaseObservation.results_updater>`
    - | Method or function that updates existing raw observation results with newly
      | gathered results.
  * - | :attr:`results_formatter <vivarium.framework.results.observation.BaseObservation.results_formatter>`
    - | Method or function that formats the raw observation results.
  * - | :attr:`stratifications <vivarium.framework.results.observation.BaseObservation.stratifications>`
    - | Optional tuple of column names for the observation to stratify by.
  * - | :attr:`to_observe <vivarium.framework.results.observation.BaseObservation.to_observe>`
    - | Method or function that determines whether to perform an observation on this Event.

The **BaseObservation** also contains the 
:meth:`observe <vivarium.framework.results.observation.BaseObservation.observe>`
method which is called at each :ref:`event <event_concept>` and :ref:`time step <time_concept>` 
to determine whether or not the observation should be recorded, and if so, gathers 
the results and stores them in the results system.

.. note::
    All four observation types discussed above inherit from the **BaseObservation** 
    abstract base class. What differentiates them are the assigned attributes 
    (e.g. defining the **results_updater** to be an adding method for the 
    **AddingObservation**) or adding other attributes as necessary (e.g. 
    adding a **stratifications**, **aggregator_sources**, and **aggregator** for
    the **StratifiedObservation**).

Stratifications
---------------

A **stratification** is a way to group and aggregate results into categories. For 
example, if you have an observation that records a certain measure but you want to 
stratify the results by age groups, you can register a stratification containing a 
mapper function that maps each simulant's age to an age group (e.g. 23.1 -> "20_to_25").

The :class:`Stratification <vivarium.framework.results.stratification.Stratification>` 
class is a data structure that holds the values and callables required to stratify the
results of an observation:

.. list-table:: **Stratification Attributes**
  :widths: 15 45
  :header-rows: 1

  * - Attribute
    - Description
  * - | :attr:`name <vivarium.framework.results.stratification.Stratification.name>`
    - | Name of the stratification.
  * - | :attr:`sources <vivarium.framework.results.stratification.Stratification.sources>`
    - | A list of the columns and values needed as input for the `mapper`.
  * - | :attr:`categories <vivarium.framework.results.stratification.Stratification.categories>`
    - | Exhaustive list of all possible stratification values.
  * - | :attr:`excluded_categories <vivarium.framework.results.stratification.Stratification.excluded_categories>`
    - | List of possible stratification values to exclude from results processing.
      | If None (the default), will use exclusions as defined in the configuration.
  * - | :attr:`mapper <vivarium.framework.results.stratification.Stratification.mapper>`
    - | A callable that maps the columns and value pipelines specified by the
      | `requires_columns` and `requires_values` arguments to the stratification
      | categories. It can either map the entire population or an individual
      | simulant. A simulation will fail if the `mapper` ever produces an invalid
      | value.
  * - | :attr:`is_vectorized <vivarium.framework.results.stratification.Stratification.is_vectorized>`
    - | True if the `mapper` function will map the entire population, and False
      | if it will only map a single simulant.

Each **Stratification** also contains the 
:meth:`stratify <vivarium.framework.results.stratification.Stratification.stratify>`
method which is called at each :ref:`event <event_concept>` and :ref:`time step <time_concept>` 
to use the **mapper** to map values in the **sources** columns to **categories** 
(excluding any categories specified in **excluded_categories**).

.. note::
    There are two types of supported stratifications: *unbinned* and *binned*;
    both types are backed by an instance of **Stratification**.
