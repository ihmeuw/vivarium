import functools
from typing import List
import textwrap

from vivarium.exceptions import VivariumError


class LifeCycleError(VivariumError):
    """Error raised when lifecycle contracts are violated."""
    pass


class ConstraintError(LifeCycleError):
    pass


class LifeCycleState:
    """A representation of the state of the simulation lifecycle."""

    def __init__(self, name: str):
        self._name = name
        self._next = None
        self._loop_next = None
        self._entrance_count = 0
        self.handlers = []

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

    def add_handlers(self, handlers):
        for h in handlers:
            name = h.__name__
            if hasattr(h, '__self__'):
                obj = h.__self__
                self.handlers.append(f'{obj.__class__.__name__}({obj.name}).{name}')
            else:
                self.handlers.append(f'Unbound function {name}')

    def __repr__(self):
        return f'LifeCycleState(name={self.name})'

    def __str__(self):
        return '\n\t'.join([self.name] + self.handlers)


class LifeCyclePhase:
    """A representation of a distinct lifecycle phase in the simulation.

    A lifecycle phase is composed of one or more distinct lifecycle states.
    There is exactly one state within the phase which serves as a valid
    exit point from the phase.  The states may operate in a loop."""

    def __init__(self, name: str, states: List[str], loop: bool):
        self._name = name
        self._next = None
        self._states = [LifeCycleState(states[0])]
        self._loop = loop
        for s in states[1:]:
            self._states.append(LifeCycleState(s))
            self._states[-2].add_next(self._states[-1])
        if self._loop:
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

    def __contains__(self, state_name):
        return bool([s for s in self._states if s.name == state_name])

    def __repr__(self):
        return f'LifeCyclePhase(name={self.name}, states={[s.name for s in self.states]})'

    def __str__(self):
        out = self.name
        if self._loop:
            out += '*'
        out += '\n' + textwrap.indent('\n'.join([str(state) for state in self.states]), '\t')
        return out


class LifeCycle:

    def __init__(self):
        self._state_names = set()
        self._phase_names = set()
        self._phases = []
        self.add_phase('initialization', ['initialization'], loop=False)

    def add_phase(self, phase_name: str, states: List[str], loop):
        """Add a new phase to the lifecycle.

        Phases must be added in order."""
        self._validate(phase_name, states)

        new_phase = LifeCyclePhase(phase_name, states, loop)
        if self._phases:
            self._phases[-1].add_next(new_phase)

        self._state_names.update(states)
        self._phase_names.add(phase_name)
        self._phases.append(new_phase)

    def get_state(self, state_name):
        if state_name not in self:
            raise LifeCycleError(f'Attempting to look up non-existent state {state_name}')
        phase = [p for p in self._phases if state_name in p].pop()
        return phase.get_state(state_name)

    def get_states(self, phase_name):
        if phase_name not in self._phase_names:
            raise LifeCycleError(f'Attempting to look up states from non-existent phase {phase_name}')
        phase = [p for p in self._phases if p.name == phase_name].pop()
        return [s.name for s in phase.states]

    def _validate(self, phase_name: str, states: List[str]):
        if phase_name in self._phase_names:
            raise LifeCycleError(f"Lifecycle phase names must be unique. You're attempting "
                                 f"to add {phase_name} but it already exists.")
        if len(states) != len(set(states)):
            raise LifeCycleError(f'Attempting to create a life cycle phase with duplicate state names. '
                                 f'States: {states}')
        duplicates = self._state_names.intersection(states)
        if duplicates:
            raise LifeCycleError(f"Lifecycle state names must be unique.  You're attempting "
                                 f"to add {duplicates} but they already exist.")

    def __contains__(self, state_name):
        return state_name in self._state_names

    def __repr__(self):
        return f'LifeCycle(phases={self._phase_names})'

    def __str__(self):
        return '\n'.join([str(phase) for phase in self._phases])


class ConstraintMaker:

    def __init__(self, lifecycle_manager):
        self.lifecycle_manager = lifecycle_manager

    def __call__(self, constrained_function, permitted_states):

        @functools.wraps(constrained_function)
        def _wrapped(*args, **kwargs):
            current_state = self.lifecycle_manager.current_state
            if current_state not in permitted_states:
                raise ConstraintError(f'Trying to call {constrained_function} during {current_state},'
                                      f' but it may only be called during {permitted_states}.')
            return constrained_function(*args, **kwargs)

        if hasattr(constrained_function, '__self__'):
            setattr(constrained_function.__self__, constrained_function.__name__, _wrapped)
        else:
            raise UnboundLocalError


class LifeCycleManager:

    def __init__(self, ):
        self.lifecycle = LifeCycle()
        self._current_state = self.lifecycle.get_state('initialization')
        self._make_constraint = ConstraintMaker(self)
        self.add_constraint(self.add_constraint, allow_during=['initialization', 'setup'])
        self.add_constraint(self.add_handlers, allow_during=['post_setup'])

    @property
    def name(self):
        return 'life_cycle_manager'

    @property
    def current_state(self):
        return self._current_state.name

    def add_phase(self, phase_name: str, states: List[str], loop: bool = False):
        self.lifecycle.add_phase(phase_name, states, loop)

    def set_state(self, state):
        new_state = self.lifecycle.get_state(state)
        if self._current_state.valid_next_state(new_state):
            new_state.enter()
            self._current_state = new_state
        else:
            raise LifeCycleError(f'Invalid transition from {self._current_state.name} to {new_state.name} requested.')

    def valid_next_state(self, state):
        return self._current_state.valid_next_state(state)

    def get_states(self, phase):
        return self.lifecycle.get_states(phase)

    def add_handlers(self, state_name, handlers):
        s = self.lifecycle.get_state(state_name)
        s.add_handlers(handlers)

    def add_constraint(self, constrained_function, allow_during=None, restrict_during=None):
        if allow_during and restrict_during or not (allow_during or restrict_during):
            raise ValueError('Must provide exactly one of "allow_during" or "restrict_during"')
        if restrict_during:
            allow_during = [s for s in self.lifecycle._state_names if s not in restrict_during]
        self._make_constraint(constrained_function, allow_during)

    def __repr__(self):
        return f'LifeCycleManager(state={self.current_state})'

    def __str__(self):
        return str(self.lifecycle)


class LifeCycleInterface:

    def __init__(self, manager: LifeCycleManager):
        self._manager = manager

    def add_handlers(self, state, handlers):
        self._manager.add_handlers(state, handlers)

    def add_constraint(self, constrained_function, allow_during=None, restrict_during=None):
        self._manager.add_constraint(constrained_function, allow_during, restrict_during)
