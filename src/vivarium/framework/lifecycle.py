from typing import List, Union, Tuple

from vivarium.exceptions import VivariumError


class LifeCycleError(VivariumError):
    """Error raised when lifecycle contracts are violated."""
    pass


class LifeCycleState:
    """A representation of the state of the simulation lifecycle."""

    def __init__(self, name: str):
        self._name = name
        self._next = None
        self._loop_next = None
        self._entrance_count = 0

    @property
    def name(self) -> str:
        """The name of the lifecycle state."""
        return self._name

    def add_next(self, next_state: 'LifeCycleState', loop: bool = False):
        if loop:
            if not self._loop_next:
                self._loop_next = next_state
            else:
                raise LifeCycleError()
        else:
            if not self._next:
                self._next = next_state
            else:
                raise LifeCycleError()

    def valid_next_state(self, state: 'LifeCycleState'):
        return state is self._next or state is self._loop_next

    def enter(self):
        self._entrance_count += 1

    def __repr__(self):
        return f'LifeCycleState(name={self.name})'

    def __str__(self):
        out = self.name
        if self._loop_next:
            out += '*'
        return out


class LifeCyclePhase:
    """A representation of a distinct lifecycle phase in the simulation.

    A lifecycle phase is composed of one or more distinct lifecycle states.
    There is exactly one state within the phase which serves as a valid
    exit point from the phase.  The states may operate in a loop."""

    def __init__(self, name: str, states: List[str], loop: bool):
        self._name = name
        self._next = None
        self._states = [LifeCycleState(states[0])]
        for s in states[1:]:
            self._states.append(LifeCycleState(s))
            self._states[-2].add_next(self._states[-1])
        if loop:
            self._states[-1].add_next(self._states[0], loop=True)

    @property
    def name(self):
        return self._name

    @property
    def states(self):
        return tuple(self._states)

    def add_next(self, phase: 'LifeCyclePhase'):
        self._next = phase
        self._states[-1].add_next(phase._states[0])

    def get_state(self, state_name: str) -> LifeCycleState:
        return [s for s in self._states if s.name == state_name].pop()

    def __repr__(self):
        return f'LifeCyclePhase(name={self.name}, states={[s.name for s in self.states]})'

    def __str__(self):
        return self.name + '\n\t' + '\n\t'.join([str(state) for state in self.states])


class LifeCycle:

    def __init__(self):
        self._phases = [LifeCyclePhase('bootstrap', ['bootstrap'], loop=False)]

    def add_phase(self, phase_name: str, states: List[str], loop):
        """Add a new phase to the lifecycle.

        Phases must be added in order."""
        new_phase = LifeCyclePhase(phase_name, states, loop)
        self._phases[-1].add_next(new_phase)
        self._phases.append(new_phase)

    def get_state(self, full_name):
        phase_name, state_name = full_name.split('.')
        phase = [p for p in self._phases if p.name == phase_name].pop()
        return phase.get_state(state_name)

    def get_states(self, phase_name):
        phase = [p for p in self._phases if p.name == phase_name].pop()
        return [s.name for s in phase.states]

    def __contains__(self, item):

    def __repr__(self):
        return f'LifeCycle(phases={[p.name for p in self._phases]})'

    def __str__(self):
        return '\n'.join([str(phase) for phase in self._phases])


class LifeCycleManager:

    def __init__(self, ):
        self.lifecycle = LifeCycle()
        self.current_state = self.lifecycle.get_state('bootstrap.bootstrap')

    @property
    def name(self):
        return 'life_cycle_manager'

    def add_phase(self, phase_name: str, states: List[str], loop: bool = False):
        self.lifecycle.add_phase(phase_name, states, loop)

    def set_state(self, state):
        new_state = self.lifecycle.get_state(state)
        if self.current_state.valid_next_state(new_state):
            new_state.enter()
            self.current_state = new_state
        else:
            raise LifeCycleError(f'Invalid transition from {self.current_state} to {new_state} requested.')

    def get_states(self, phase):
        return self.lifecycle.get_states(phase)

    def __repr__(self):
        return f'LifeCycleManager(state={self.current_state})'

    def __str__(self):
        return str(self.lifecycle)


class LifeCycleInterface:

    def __init__(self, manager: LifeCycleManager):
        self._manager = manager

    def add_handlers(self, state, handlers):
        pass



