.. _crn_concept:

=====================
Common Random Numbers
=====================

.. contents::
   :depth: 2
   :local:
   :backlinks: none

Outline
-------

- Variance reduction
- Counterfactual Analysis


Why randomness?
let's you ensure that the variations you see when running e.g., counterfactuals are due only to changes you made and don't
include other noise

in a very literal sense, it means that John Smith in simulation A lives exactly the same life as John Smith in simulation
B except for decisions that are informed by the intervention that makes B the counterfactual of A. To illustrate, say we
are interested in an intervention that reduces traffic accidents. Simulation B should be identical to simulation A except
that fewer traffic accidents occur. John wakes up at the same time. He decides to eat the same breakfast. He leaves the
house at the same time. He decides to take the same route to work. In simulation A, he gets into a fender bender on the
way to work. In simulation B, because we are explicitly reducing the likelihood of traffic accidents, perhaps he does not.


What does that mean from a practical standpoint? For users, it means we can be certain that the differences we see between
a baseline simulation and a counterfactual are due only to the changes we made - only to what we said made the counterfactual
different from the baseline.

It also means, however, that we need some way of identifying simulants that's independent of the simulation. We need
John Smith to always be John Smith. For simulations without any method of adding simulants except during the initialization
of the simulation, this is easy to think of - if John Smith is person 500 (that is the 500th simulant initialized in the baseline
simulation), person 500 in the counterfactual will also be John Smith. But what if our 'intervention' (that is, what makes
the counterfactual different from the baseline) concerns increasing fertility?

Users determine what uniquely identifies simulants across simulations. It defaults to entrance time.

Let's say sim A is baseline and sim B is counterfactual. Our intervention means that fertility rates are twice as high
in sim B than sim A. We initialize both simulations to start with a population of 300 people. In sim A, let's say it
takes us 2 years to get to 500 people - that is, John will be born 2 years after the beginning of the simulation. But in
sim B, the 500th person born will be born after 1 year. If we say that entrance time identifies simulants, that person is
not John. John is the simulant who enters the simulation 2 years after the beginning of the simulation. In the counterfactual,
that's probably more like the 1000th person.


SO what should you care about from this:
That you have a decision to  make. You need to decide what will identify simulants across your baseline and counterfactual
simulations.

include pictures of cube and lining up sims here


also what lets you run multiple small sims and aggregate up?