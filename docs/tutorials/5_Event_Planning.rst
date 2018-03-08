Tutorial Five: Event Planning
=============================

Event's are integral to Vivarium but up to this point we've only talked about a couple of standard events emitted by the framework itself. We can emit our own events and that's key to some kinds of complex models. In this tutorial we'll talk about emitting and responding to custom events.

Emitting
--------

Right now the only thing that really happens in our model is that people die, so let's attach our custom event to that. We'll need to change the Mortality component to emit the event whenever simulants die. There is a decorator equivalent called `emits` which, when applied to a function, will inject an event emitter object into it's arguments when it is called. We import it from `vivarium.framework.event` just like the `listens_for` decorator:

.. code-block:: python

    from vivarium.framework.event import listens_for, emits

Then we apply it to the method where deaths actually happen, supplying it with a name for the new event type and adding a parameter to accept the emitter object:

.. code-block:: python

        @listens_for('time_step')
        @emits('death')
        @uses_columns(['alive'], 'alive == True')
        def handler(self, event, emitter):
            effective_rate = self.mortality_rate(event.index)
            effective_probability = 1-np.exp(-effective_rate)
            draw = np.random.random(size=len(event.index))
            affected_simulants = draw < effective_probability
            event.population_view.update(pd.Series(False, index=event.index[affected_simulants]))



Now that we have an emitter we need to actually use it. To do that we need to construct a new event to send. The easiest way to make new events is by copying an existing event and supplying it with a new index. You can do that with the `event.split` method:

.. code-block:: python

            new_event = event.split(event.index[affected_simulants])

That gives us a new event that applies to the same time as the current event, which is what we want, but only applies to the subset of the population who just died. We can then send it out to any listeners using the emitter:

.. code-block:: python

            emitter(new_event)


Listening
---------

With those changes, the events will go out but nothing will actually happen because there isn't anyone listening for them yet. There are lots of things we might do with information about who died but let's imagine that that we're interested in tracking funeral costs.

.. code-block:: python

    class FuneralCosts:
        def setup(self, builder):
            self.costs = 0

        @listens_for('simulation_end')
        def report(self, event):
            print("Total funeral costs: {}".format(self.costs))

Here we have a new component with a simple counter and a reporting function analogous to what we have in the `Mortality` component. To add costs to the counter we'll need to listen for the death events we're now emitting. That's the same as listening for `time_step` and other events we're already using, just with a different name.

.. code-block:: python

        @listens_for('death')
        def count_costs(self, event):

Now that we have the event we can find out how many people died by looking at the size of the index, which we know will only contain the recently desciesed.

.. code-block:: python

            deaths = len(event.index)
            self.costs += deaths * 6078 # average cost of a cremation in 2105 according to the NFDA

And that's it. Hopefully that makes the basic pattern for using events clear.
