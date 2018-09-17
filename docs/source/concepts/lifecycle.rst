The Simulation Lifecycle
========================

.. contents::
   :depth: 2
   :local:
   :backlinks: none

Outline
-------

- Entry point variations (link to more detailed guide)
- Initialization (create objects, compile configuration)
- Setup (Wire components, load data)
- Post setup
- Population Initialization
- Event loop
  - time_step__prepare
  - time_step
  - time_step__cleanup
  - collect_metrics
- simulation_end
