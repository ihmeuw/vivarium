from code import interact

from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns

@listens_for('time_step')
@uses_columns(None)
def inspect(event):
    interact(banner="""
Inspecting population at {}.
Population table in 'population'.
To stop the simulation use 'exit()'.
To continue to the next time step use 'Ctrl-D' ('Ctrl-Z' on Windows).
""".format(event.time), local={'population':event.population})
