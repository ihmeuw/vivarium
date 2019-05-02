import inspect
from typing import Sequence, Any

from vivarium import VivariumError, ConfigTree


class ComponentConfigError(VivariumError):
    """Error while interpreting configuration file or initializing components"""
    pass


class ComponentManager:
    """Responsible for tracking all components in a Vivarium simulation, enabling
    access to them, and for initiating the ``setup`` life-cycle stage of each
    component."""

    def __init__(self):
        self._managers = []
        self._components = []

    def add_managers(self, managers: Sequence):
        """Registers new managers with the system. Managers are setup before
        components.

        Parameters
        ----------
        managers:
          Instantiated managers to register
        """
        self._add_components(self._managers, managers)

    def add_components(self, components: Sequence):
        """Register new components with the system. Components are setup after
        managers.

        Parameters
        ----------
        components:
          Instantiated components to register
        """
        self._add_components(self._components, components)

    def _add_components(self, component_list: Sequence, components: Sequence):
        for component in components:
            if isinstance(component, Sequence):
                self._add_components(component_list, component)
            else:
                component_list.append(component)

    def get_components(self, component_type: Any) -> Sequence:
        """Return a list of components that are instances of a certain type
        currently maintained by  the component manager. Does not include other
        managers.

        Parameters
        ----------
        component_type
            A component class.
        Returns
        -------
            A list of components.
        """
        return [c for c in self._components if isinstance(c, component_type)]

    def setup_components(self, builder, configuration: ConfigTree):
        """Apply component-level configuration defaults to the global configuration
        and run setup methods. Runs first on managers, then on components.

        This can result in new components due to side effects of setup.

        Parameters
        ----------
        builder:
            Interface to several simulation tools.
        configuration:
            Simulation configuration parameters.
        """
        self._managers = _setup_components(builder, self._managers, configuration)
        self._components = _setup_components(builder, self._components, configuration)


class ComponentInterface:
    """The component manager interface available from the builder."""
    def __init__(self, component_manager: ComponentManager):
        self._component_manager = component_manager

    def add_components(self, components: Sequence):
        """Register new components with the system. Components are setup after
        managers.

        Parameters
        ----------
        components:
          Instantiated components to register
        """
        self._component_manager.add_components(components)

    def get_components(self, component_type: str):
        return self._component_manager.get_components(component_type)


def _setup_components(builder, component_list: Sequence, configuration: ConfigTree) -> Sequence:

    configured = []
    for c in component_list:  # apply top-level configurations first
        if hasattr(c, "configuration_defaults") and not c in configured:
            _apply_component_default_configuration(configuration, c)
            configured.append(c)

    setup = []
    while component_list:  # mutated at runtime by calls to setup
        c = component_list.pop(0)
        if c is None:
            raise ComponentConfigError('None in component list. This likely '
                                       'indicates a bug in a factory function')

        if hasattr(c, "configuration_defaults") and c not in configured:
            _apply_component_default_configuration(configuration, c)
            configured.append(c)

        if c not in setup:
            if hasattr(c, "setup"):
                c.setup(builder)
            setup.append(c)

    return setup


def _apply_component_default_configuration(configuration, component):
    # This reapplies configuration from some components but it is idempotent so there's no effect.
    if component.__module__ == '__main__':
        # This is defined directly in a script or notebook so there's no file to attribute it to
        source = '__main__'
    else:
        source = inspect.getfile(component.__class__)
    _check_duplicated_default_configuration(component.configuration_defaults, configuration, source)
    configuration.update(component.configuration_defaults, layer='component_configs', source=source)


def _check_duplicated_default_configuration(component, config, source):
    overlapped = set(component.keys()).intersection(config.keys())
    if not overlapped:
        pass

    while overlapped:
        key = overlapped.pop()

        try:
            sub_config = config.get_from_layer(key, layer='component_configs')
            sub_component = component[key]

            if isinstance(sub_component, dict) and isinstance(sub_config, ConfigTree):
                _check_duplicated_default_configuration(sub_component, sub_config, source)
            elif isinstance(sub_component, dict) or isinstance(sub_config, ConfigTree):
                raise ComponentConfigError(f'These two sources have different structure of configurations for {component}.'
                                           f' Check {source} and {sub_config}')
            else:
                raise ComponentConfigError(f'Check these two {source} and {config._children[key].get_value_with_source()}'
                                           f'Both try to set the default configurations for {component}/{key}')

        except KeyError:
            pass



