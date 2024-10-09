**3.0.11 - 10/08/24**

  - Fix mypy errors: vivarium/framework/values.py

**3.0.10 - 10/07/24**

  - Add method to SimulationClock to get number of simulation steps remaining

**3.0.9 - 10/04/24**

  - Fix mypy errors: vivarium/framework/logging/utilities.py

**3.0.8 - 10/03/24**

  - Fix mypy errors: vivarium/framework/resource.py
  - Fix mypy errors: vivarium/framework/artifact/hdf.py

**3.0.7 - 09/25/24**

  - Enable population manager and population view methods to accept strings  
  - Fix mypy errors: vivarium/framework/lifecycle.py
  - Fix mypy errors: vivarium/framework/population/manager.py
  - Fix mypy errors: vivarium/framework/population/population_view.p
  - Fix mypy errors: vivarium/framework/plugins.py
  - Fix mypy errors: vivarium/framework/results/stratification.py

**3.0.6 - 09/20/24**

  - Fix mypy errors: vivarium/framework/randomness/index_map.py

**3.0.5 - 09/17/24**

  - Pin Sphinx below 8.0

**3.0.4 - 09/12/24**

  - Introduce static type checking with mypy
  - Add new types for clock time and step-size

**3.0.3 - 09/11/24**

  - Raise an error if a component attempts to access a non-existent population_view

**3.0.2 - 08/27/24**

  - Update results docstrings
  - Add a results concepts model doc
  - Docstring cleanup throughout
  - Fix up tutorial docs
  - Strengthen doctests
  
**3.0.1- 08/20/24**

 - Create script to find matching dependency branches
 - Add results category exclusion tests

**3.0.0 - 08/13/24**

Breaking changes:
  - Remove the unnecessary metrics pipeline
  - Refactor lookup table creation and allow configuration of lookup columns
  - Refactor results manager to process results directly

Major changes:
  - Move results controller and interface from managers to default plugins
  - Add a get_configuration method and configuration attribute to Component
  - Enable build_lookup_table to accept a list as input data
  - Implement an Observation dataclass
  - Remove --location/-l option from simulate run command
  - Change the metrics/ folder to results/; refer to "results" instead of "metrics" throughout
  - Implement multiple new interface functions for registering stratifications and observations
  - Implement multiple new Observer types
  - Implement simulation backups
  - Implement excluding results categories via the configuration

Other changes:
  - Use layered-config-tree package instead of local ConfigTree class
  - Add a report emitter to the SimulationContext
  - Check for and log unused stratifications and raise for missing required stratifications
  - Move all steps of running a simulation into a run_simulation instance method
  - Add simulate run e2e test
  - Stop writing seed and draw columns to the results
  - Install matching upstream branches in github builds
  - Automatically run Jenkins builds on push or pull request
  - Add type hints throughout results modules
  - Various other minor updates

**2.3.8 - 06/17/24**

 - Hotfix pin numpy below 2.0

**2.3.7 - 03/21/24**
  
  - Add deprecation warning to import ConfigTree from the config_tree package

**2.3.6 - 03/07/24**

  - Update population keys in testing utilities to be mmore descriptive

**2.3.5 - 03/01/24**

  - Improves boids example and tutorial

**2.3.4 - 02/23/24**

  - Fixes parsing in results manager to remove trailing underscore

**2.3.3 - 01/29/24**

 - Improve readability of api reference docs

**2.3.2 - 01/29/24**

 - Fix broken readthedocs build

**2.3.1 - 01/09/24**

 - Update PyPI to 2FA with trusted publisher

**2.3.0 - 12/19/23**

 - Incorporate Individualized Clocks v1
 - Document how to remove components from an interactive simulation
 - Update code in docs to match current implementation

**2.2.1 - 10/24/23**

 - Hotfix to expose ScalarValue at the lookup package level

**2.2.0 - 10/24/23**

 - Refactor Manager configuration defaults
 - Throw an error if simulation attempts to use a component that is not an instance of Component
 - Refactor and simplify LookupTable implementations
 - Enable LookupTable to have source data that is only categorical
 - Enable LookupTables with DataFrame source data to specify value columns

**2.1.1 - 10/13/23**

 - Enable RandomnessStream to sample from a distribution
 - Refactor `ComponentConfigurationParser` to create components as they are parsed

**2.1.0 - 10/12/23**

 - Remove explicit support for Python 3.8
 - Set default value for ConfigNode::get_value to None

**2.0.1 - 10/04/23**

 - Set pipeline's returned Series' name to the pipeline's name

**2.0.0 - 09/22/23**

 - Create `Component` and `Manager` classes
 - Ensure all managers and components inherit from them
 - Define properties in `Component` that components can override
 - Define lifecycle methods in `Component` that components override
 - Refactor all components in vivarium to use and leverage`Component`
 - Warn when using components not inheriting from `Component`
 - Change the behavior of `State.add_transition` to take a Transition object rather than another State
 - Add optional argument to State constructor to allow self transitions

**1.2.9 - 09/19/23**

 - Set default in register_observation

**1.2.8 - 09/18/23**

 - Unpin pandas

**1.2.7 - 09/14/23**

 - Allow pandas <2.1.0

**1.2.6 - 09/14/23**

 - Update state machine to prepare for pandas 2.0

**1.2.5 - 09/05/23**

 - Update ConfigTree to make it pickleable; raise NotImplementedError on equality calls

**1.2.4 - 09/01/23**

 - Create LookupTableData type alias for the source data to LookupTables

**1.2.3 - 08/28/23**

 - Enable allowing self transitions directly in a State's constructor

**1.2.2 - 08/04/23**

 - Bugfix to include all metrics outputs in results manager

**1.2.1 - 07/12/23**

 - Adds logging for registering stratifications and observations
 - Changes version metadata to use setuptools_scm

**1.2.0 - 06/01/23**

 - Stop supporting Python 3.7 and start supporting 3.11
 - Bugfix to allow for zero stratifications
 - Removes ignore filters for known FutureWarnings
 - Refactor location of default stratification definition
 - Bugfix to stop shuffling simulants when drawing common random number

**1.1.0 - 05/03/23**

 - Clean up randomness system
 - Fix a bug in stratification when a stratum is empty
 - Create a dedicated logging system
 - Fix bug in preventing passing an Iterable to `rate_to_probability`

**1.0.4 - 01/25/23**

 - Bugfixes for ResultsContext

**1.0.3 - 01/19/23**

 - Enhancement to use pop_data.user_data.get pattern in BasePopulation example
 - Mend get_value unhashable argument for Results Manger add_observation()
 - Split randomness into subpackage
 - Remove copy_with_additional_key method from RandomnessStream

**1.0.2 - 12/27/22**

 - Fix a typo that prevented deployment of v1.0.1

**1.0.1 - 12/27/22**

 - Remove metrics from the population management system
 - Add a new lifecycle builder interface method for simulation state access
 - Suppress future warnings (temporarily)
 - Update github actions to support python 3.7-3.10
 - Update codeowners

**1.0.0 - 12/20/22**

 - Added Results Manager feature.

**0.10.21 - 12/20/22**

 - Cleaned up warnings in artifact test code.
 - Updated codeowners and pull request template.

**0.10.20 - 12/20/22**

 - Update CI versions to build on python versions 3.7-3.10

**0.10.19 - 10/04/22**

 - Fix bug on `simulate run` CLI introduced in 0.10.18

**0.10.18 - 09/20/22**

 - Standardize results directories
 - Adds ability to run without artifact
 - Specify correct permissions when creating directories and files

**0.10.17 - 07/25/22**

 - Fix bug when initializing tracked column

**0.10.16 - 06/30/22**

 - Fix a bug in adding new simulants to a population
 - Add CODEOWNERS file

**0.10.15 - 06/29/22**

 - Added performance reporting
 - Added support for empty initial populations
 - Refactor population system

**0.10.14 - 05/16/22**

 - Fixed pandas FutureWarning in `randomness.get_draw`

**0.10.13 - 05/05/22**

 - Improved error message when component dependencies are not specified.
 - Fix faulty set logic in `PopulationView.subview`

**0.10.12 - 02/15/22**

 - Reformat code with black and isort.
 - Add formatting checks to CI.
 - Add `current_time` to interactive context.
 - Squash pandas FutureWarning for Series.append usage.
 - Add a UserWarning when making a new artifact.

**0.10.11 - 02/12/22**

 - Update CI to make a cleaner release workflow
 - Add PR template

**0.10.10 - 10/29/21**

 - Update license to BSD 3-clause
 - Replace authors metadata with zenodo.json
 - Updated examples
 - Doctest bugfixes

**0.10.9 - 08/16/21**

 - Add flag to SimulationContext.report to turn off results printing at sim end.

**0.10.8 - 08/10/21**

 - Set Python version in CI deployment to 3.8

**0.10.7 - 08/10/21**

 - Hotfix to re-trigger CI

**0.10.6 - 08/10/21**

 - Fix bug in deploy script

**0.10.5 - 08/10/21**

 - Update builder documentation
 - Update build process
 - Add check for compatible python version

**0.10.4 - 04/30/21**

 - Reapply location and artifact path changes

**0.10.3 - 04/30/21**

 - Revert location and artifact path changes

**0.10.2 - 04/27/21**

 - Remove dependency on location and artifact path in configuration files
 - Add location and artifact path arguments to `simulate run`
 - Fix bug that broke simulations running on Windows systems

**0.10.1 - 12/24/20**

 - Move from travis to github actions for CI.

**0.10.0 - 10/2/20**

 - Fix bug in copying a `RandomnessStream` with a new key
 - Add documentation of randomness in vivarium
 - Add validation to `LookupTable`, `InterpolatedTable`, `Interpolation`, and
   `Order0Interp`
 - Fix bug writing invalid artifact keys
 - Fix `EntityKey` `eq` and `ne` functions
 - Remove dependency on `graphviz`
 - Move `get_seed` from `RandomnessStream` to `RandomnessInterface`
 - Remove `random_seed` from output index and add `random_seed` and
   `input_draw` to output columns
 - Raise a `PopulationError` when trying to access non-existent columns in a
   `PopulationView`
 - Fix validation issues in Travis config
 - Fix typing issues in `ComponentManager` and `Event`

**0.9.3 - 12/7/19**

 - Bugfix in population type conversion.

**0.9.2 - 12/3/19**

 - Bugfix in artifact configuration management.
 - Bugfix in population query.

**0.9.1 - 11/18/19**

 - Be less restrictive about when get_value can be called.

**0.9.0 - 11/16/19**

 - Clean up event emission.
 - Make events immutable.
 - Stronger validation around model specification file.
 - Move the data artifact from vivarium public health to vivarium.
 - Update the ConfigTree str and repr to be more legible.
 - Be consistent about preferring pathlib over os.path.
 - Add some ConfigTree specific errors.
 - Refactor ConfigTree and ConfigNode to remove unused functionality and
   make the interface more consistent.
 - Extensively update documentation for configuration system.
 - Restructure component initialization so that **all** simulation components
   are created at simulation initialization time. Previous behavior had
   sub-components created at setup time.
 - Introduce lifecycle management system to enforce events proceed in the
   correct order and ensure framework tools are not misused.
 - Remove results writer.
 - Overhaul simulation creation to be significantly less complex.
 - Update privacy levels for simulation context managers.
 - Update context creation and usage tutorials.
 - Ditch the 'omit_missing_columns' argument for PopulationView.get.  Subviews
   should be used instead.
 - Consistent naming for rates in data, pipelines, and configuration.
 - Introduce resource management system for users to properly specify
   component dependencies for population initialization.
 - Switch age_group_start and age_group_end to age_start and age_end, making
   the naming scheme for binned data consistent.
 - Use loguru for logging.
 - Fix a bug in transition probability computation.
 - Raise error when component attempts to update columns they don't own instead
   of silently ignoring them.
 - Use consistent data bin naming to make using lookup tables less verbose.
 - Rename value system joint_value_postprocessor to union_postprocessor.
 - Docs and concept note for values system.
 - Be consistent about manager naming on builder interfaces.
 - Updated concept docs for entry points.
 - Lookup table docs and concept note.
 - Bugfix in randomness to handle datetime conversion on Windows.
 - Constrain components to only have a single population initializer.

**0.8.24 - 08/20/19**

 - Bugfix to prevent component list from not including setup components during setup phase.
 - Bugfix to dot diagram of state machine.

**0.8.23 - 08/09/19**

 - Move handle_exceptions() up to vivarium to eliminate duplication

**0.8.22 - 07/16/19**

 - Bugfix for lookup table input validation.
 - Event subsystem documentation.

**0.8.21 - 06/14/19**

 - Add names and better reprs to some of the managers.
 - ConfigTree documentation
 - Yaml load bugfix.
 - Documentation for ``simulate run`` and the interactive context.
 - Tutorials for running a simulation interactively and from the command line.
 - Headers for API documentation.
 - Component management documentation.
 - Enforce all components have a unique name.
 - Add ``get_components_by_type`` and ``get_component(name)`` to
   the component manager.
 - Bugfix in the lookup table.

**0.8.20 - 04/22/19**

 - Add simulation lifecycle info to the simulant creator.
 - Bugfix in simulate profile.

**0.8.19 - 03/27/19**

 - Update results writer to write new hdfs instead of overwriting.

**0.8.18 - 02/13/19**

 - Fix numerical issue in rate to probability calculation
 - Alter randomness manager to keep track of randomness streams.

**0.8.17 - 02/13/19**

 - Fix branch/version synchronization

**0.8.16 - 02/11/19**

 - Remove combined sexes from the "build_table".

**0.8.15 - 01/03/19**

 - Add doctests to travis
 - Update population initializer error message

**0.8.14 - 12/20/18**

 - Standardize the population getter from the the interactive interface.
 - Added "additional_key" argument to randomness.filter for probability and for rate.
 - Added a profile subcommand to simulate.
 - Separated component configuration from setup.
 - Vectorize python loops in the interpolation implementation.

**0.8.13 - 11/15/18**

 - Fix broken doc dependency

**0.8.12 - 11/15/18**

 - Remove mean age and year columns

**0.8.11 - 11/15/18**

 - Bugfix where transitions were casting pandas indices to series.
 - Add better error message when a none is found in the configuration.

**0.8.10 - 11/5/18**

 - Added ``add_components`` method to simulation context.
 - Added typing info to interactive interface.

**0.8.9 - 10/23/18**

 - Accept ``.yml`` model specifications
 - Redesign interpolation. Order zero only at this point.

**0.8.8 - 10/09/18**

 - Raise error if multiple components set same default configuration.
 - Loosen error checking in value manager

**0.8.7 - 09/25/18**

 - Distinguish between missing and cyclic population table dependencies.
 - Initial draft of tutorial documentation

**0.8.6 - 09/07/18**

 - Workaround for hdf metadata limitation when writing dataframes with a large
   number of columns

**0.8.5 - 08/22/18**

 - Add integration with Zenodo to produce DOIs
 - Added default get_components implementation for component manager

**0.8.4 - 08/02/18**

 - Standardized a bunch of packaging stuff.

**0.8.2 - 07/24/18**

 - Added ``test`` command to verify and installation
 - Updated ``README`` with installation instructions.


**0.8.1 - 07/24/18**

 - Move to source layout.
 - Set tests to install first and then test installed package.
 - Renamed ``test_util`` to resolve naming collision during test.

**0.8.0 - 07/24/18**

 - Initial release
