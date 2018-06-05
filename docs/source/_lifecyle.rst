Stuff
=====

This is the sequence of life-cycle stages of the simulation:

.. graphviz::

     digraph lifecycle {
        compound=true;

        subgraph cluster0 {
            label="Time Step";

            time_step__prepare;
            time_step;
            time_step__cleanup;
            collect_metrics;
        }

        setup -> initialize_simulants -> time_step__prepare[lhead=cluster0];
        time_step__prepare -> time_step -> time_step__cleanup -> collect_metrics -> time_step__prepare;
        collect_metrics -> finalize [ltail=cluster0];
        finalize-> report;
     }
