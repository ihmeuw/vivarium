"""A configuration structure which supports cascading layers.

In Vivarium it allows base configurations to be overridden by component level
configurations which are in turn overridden by model level configuration
which can be overridden by user supplied overrides. From the perspective
of normal client code the cascading is hidden and configuration values
are presented as attributes of the configuration object the values of
which are the value of that key in the outermost layer of configuration
where it appears.

For example:

.. code-block:: python

    >>> config = ConfigTree(layers=['inner_layer', 'middle_layer', 'outer_layer', 'user_overrides'])
    >>> config.read_dict({'section_a': {'item1': 'value1', 'item2': 'value2'}, 'section_b': {'item1': 'value3'}}, layer='inner_layer')
    >>> config.read_dict({'section_a': {'item1': 'value4'}, 'section_b': {'item1': 'value5'}}, layer='middle_layer')
    >>> config.read_dict({'section_b': {'item1': 'value6'}}, layer='outer_layer')
    >>> config.section_a.item1
    'value4'
    >>> config.section_a.item2
    'value2'
    >>> config.section_b.item1
    'value6'
    >>> config.section_b.item1 = 'value7'
    >>> config.section_b.item1
    'value7'
"""
from typing import Mapping, Union
import yaml


class ConfigNode:
    """A single configuration value which may have different variants for different layers.
    Each ConfigNode also records the source (if any is reported) from which the value is derived.
    """
    def __init__(self, layers=None):
        if not layers:
            self._layers = ['base']
        else:
            self._layers = layers
        self._values = {}
        self._frozen = False
        self._accessed = False

    def freeze(self):
        """Causes the node to become read only. This is useful for loading and then
        freezing configurations that should not be modified at runtime.
        """
        self._frozen = True

    def get_value_with_source(self, layer=None):
        """Returns a tuple of the value's source and the value at the specified
        layer. If no layer is specified then the outer layer is used.

        Parameters
        ----------
        layer : str
            Name of the layer to use. If None then the outermost where the value
            exists will be used.

        Raises
        ------
        KeyError
            If the value is not set for the specified layer
        """
        if layer:
            return self._values[layer]

        for layer in reversed(self._layers):
            if layer in self._values:
                return self._values[layer]
        raise KeyError(layer)

    def is_empty(self):
        return not self._values

    def get_value(self, layer=None):
        """Returns the value at the specified layer.

        Parameters
        ----------
        layer : str
            Name of the layer to use. If None then the outermost where the value
            exists will be used.

        Raises
        ------
        KeyError
            If the value is not set for the specified layer
        """
        self._accessed = True
        return self.get_value_with_source(layer)[1]

    def has_been_accessed(self):
        """Returns whether this node has been accessed.

        Returns
        -------
        bool
            Whether this node has been accessed
        """
        return self._accessed

    def metadata(self):
        """Returns all values and associated metadata for this node as a dict.
        The value which would be selected if the node's value was requested
        is indicated by the `default` flag.
        """
        result = []
        for layer in self._layers:
            if layer in self._values:
                result.append({
                    'layer': layer,
                    'value': self._values[layer][1],
                    'source': self._values[layer][0],
                    'default': layer == self._layers[-1]
                })
        return result

    def set_value(self, value, layer=None, source=None):
        """Set a value for a particular layer with optional metadata about source.

        Parameters
        ----------
        value : str
            Data to store in the node.
        layer : str
            Name of the layer to use. If None then the outermost where the value
            exists will be used.
        source : str
            Metadata indicating the source of this value (e.g. a file path)

        Raises
        ------
        TypeError
            If the node is frozen
        KeyError
            If the named layer does not exist
        """
        if self._frozen:
            raise TypeError('Frozen ConfigNode does not support assignment')

        if not layer:
            layer = self._layers[-1]
        self._values[layer] = (source, value)

    def drop_layer(self, layer):
        """Removes the named layer and the value associated with it from the node.

        Parameters
        ----------
        layer : str
            Name of the layer to drop.

        Raises
        ------
        TypeError
            If the node is frozen
        KeyError
            If the named layer does not exist
        """
        if self._frozen:
            raise TypeError('Frozen ConfigNode does not support modification')
        self.reset_layer(layer)
        self._layers.remove(layer)

    def reset_layer(self, layer):
        """Removes any value and metadata associated with the named layer.

        Parameters
        ----------
        layer : str
            Name of the layer to reset.

        Raises
        ------
        TypeError
            If the node is frozen
        KeyError
            If the named layer does not exist
        """
        if self._frozen:
            raise TypeError('Frozen ConfigNode does not support modification')
        if layer in self._values:
            del self._values[layer]

    def __repr__(self):
        return 'ConfigNode(layers={}, values={}, frozen={}, accessed={})'.format(
            self._layers, self._values, self._frozen, self._accessed)

    def __str__(self):
        return '\n'.join(reversed(['{}: {}\n    source: {}'.format(layer, value[1], value[0])
                                   for layer, value in self._values.items()]))


class ConfigTree:
    """A container for configuration information. Each configuration value is
    exposed as an attribute the value of which is determined by the outermost
    layer which has the key defined.
    """
    def __init__(self, data=None, layers=None):
        """
        Parameters
        ----------
        data : dict, str, or ConfigTree, optional
            A dictionary containing initial values
        layers : list
            A list of layer names. The order in which layers defined determines
            how they cascade. Later layers override the values from earlier ones.
        """
        if not layers:
            self.__dict__['_layers'] = ['base']
        else:
            self.__dict__['_layers'] = layers
        self.__dict__['_children'] = {}
        self.__dict__['_frozen'] = False

        if data:
            self.update(data, layer=self._layers[0], source='initial data')

    def freeze(self):
        """Causes the ConfigTree to become read only.

        This is useful for loading and then freezing configurations that should not be modified at runtime.
        """
        self.__dict__['_frozen'] = True
        for child in self._children.values():
            child.freeze()

    def __setattr__(self, name, value):
        """Set a configuration value on the outermost layer."""
        self._set_with_metadata(name, value, layer=None, source=None)

    def __setitem__(self, name, value):
        """Set a configuration value on the outermost layer."""
        self._set_with_metadata(name, value, layer=None, source=None)

    def __getattr__(self, name):
        """Get a configuration value from the outermost layer in which it appears."""
        try:
            return self.get_from_layer(name)
        except KeyError:
            raise AttributeError(name)

    def __getitem__(self, name):
        """Get a configuration value from the outermost layer in which it appears."""
        return self.get_from_layer(name)

    def __delattr__(self, name):
        if name in self._children:
            del self._children[name]

    def __delitem__(self, name):
        if name in self._children:
            del self._children[name]

    def __contains__(self, name):
        """Test if a configuration key exists in any layer."""
        return name in self._children

    def __iter__(self):
        """Dictionary-like iteration."""
        return iter(self._children)

    def items(self):
        """Return a list of all (child_name, child) pairs."""
        return self._children.items()

    def keys(self):
        return self._children.keys()

    def get_from_layer(self, name, layer=None):
        """Get a configuration value from the named layer.

        Parameters
        ----------
        name : str
            The name of the value to retrieve
        layer: str
            The name of the layer to retrieve the value from. If it is not supplied
            then the outermost layer in which the key is defined will be used.
        """
        if name not in self._children:
            if self._frozen:
                raise KeyError(name)
            self._children[name] = ConfigTree(layers=self._layers)
        child = self._children[name]
        if isinstance(child, ConfigNode):
            return child.get_value(layer)
        else:
            return child

    def _set_with_metadata(self, name, value, layer=None, source=None):
        """Set a value in the named layer with the given source.

        Parameters
        ----------
        name : str
            The name of the value
        value
            The value to store
        layer : str, optional
            The name of the layer to store the value in. If none is supplied
            then the value will be stored in the outermost layer.
        source : str, optional
            The source to attribute the value to.

        Raises
        ------
        TypeError
            if the ConfigTree is frozen
        """

        if self._frozen:
            raise TypeError('Frozen ConfigTree does not support assignment')

        if isinstance(value, dict):
            if name not in self._children or not isinstance(self._children[name], ConfigTree):
                self._children[name] = ConfigTree(layers=list(self._layers))
            self._children[name].update(value, layer, source)
        else:
            if name not in self._children or not isinstance(self._children[name], ConfigNode):
                self._children[name] = ConfigNode(list(self._layers))
            child = self._children[name]
            child.set_value(value, layer, source)

    def update(self, data: Union[Mapping, str, bytes], layer: str=None, source: str=None):
        """Adds additional data into the ConfigTree.


        Parameters
        ----------
        data :
            source data
        layer :
            layer to load data into. If none is supplied the outermost one is used
        source :
            Source to attribute the values to

        See Also
        --------
        read_dict
        """
        if isinstance(data, dict):
            self._read_dict(data, layer, source)
        elif isinstance(data, ConfigTree):
            # TODO: set this to parse the other config tree including layer and source info.  Maybe.
            self._read_dict(data.to_dict(), layer, source)
        elif isinstance(data, str):
            if data.endswith('.yaml'):
                source = source if source else data
                self._load(data, layer, source)
            else:
                self._loads(data, layer, source)
        elif data is None:
            pass
        else:
            raise ValueError(f"Update must be called with dictionary, string, or ConfigTree. "
                             f"You passed in {type(data)}")

    def _read_dict(self, data_dict, layer=None, source=None):
        """Load a dictionary into the ConfigTree. If the dict contains nested dicts
        then the values will be added recursively. See module docstring for example code.

        Parameters
        ----------
        data_dict : dict
            source data
        layer : str
            layer to load data into. If none is supplied the outermost one is used
        source : str
            Source to attribute the values to
        """
        for k, v in data_dict.items():
            self._set_with_metadata(k, v, layer, source)

    def _loads(self, data_string, layer=None, source=None):
        """Load data from a yaml formatted string.

        Parameters
        ----------
        data_string : str
            yaml formatted string. The root element of the document should be
            an associative array
        layer : str
            layer to load data into. If none is supplied the outermost one is used
        source : str
            Source to attribute the values to
        """
        data_dict = yaml.load(data_string)
        self._read_dict(data_dict, layer, source)

    def _load(self, f, layer=None, source=None):
        """Load data from a yaml formatted file.

        Parameters
        ----------
        f : str or file like object
            If f is a string then it is interpreted as a path to the file to load
            If it is a file like object then data is read directly from it.
        layer : str
            layer to load data into. If none is supplied the outermost one is used
        source : str
            Source to attribute the values to
        """
        if hasattr(f, 'read'):
            self._loads(f.read(), layer=layer, source=source)
        else:
            with open(f) as f:
                self._loads(f.read(), layer=layer, source=source)

    def to_dict(self):
        result = {}
        for k, v in self._children.items():
            if isinstance(v, ConfigNode):
                result[k] = v.get_value()
            else:
                result[k] = v.to_dict()
        return result

    def metadata(self, name):
        """Return value and metadata associated with the named value

        Parameters
        ----------
        name : str
            name to retrieve. If the name contains '.'s it will be retrieved recursively

        Raises
        ------
        KeyError
            if name is not defined in the ConfigTree
        """
        if name in self._children:
            return self._children[name].metadata()
        else:
            head, _, tail = name.partition('.')
            if head in self._children:
                return self._children[head].metadata(key=tail)
            else:
                raise KeyError(name)

    def reset_layer(self, layer, preserve_keys=()):
        """Removes any value and metadata associated with the named layer.

        Parameters
        ----------
        layer : str
            Name of the layer to reset.
        preserve_keys : list or tuple
            A list of keys to skip while removing data

        Raises
        ------
        TypeError
            If the node is frozen
        KeyError
            If the named layer does not exist
        """
        self._reset_layer(layer, [k.split('.') for k in preserve_keys], prefix=[])

    def _reset_layer(self, layer, preserve_keys, prefix):
        if self._frozen:
            raise TypeError('Frozen ConfigTree does not support modification')
        deletable = []
        for key, child in self._children.items():
            if prefix + [key] not in preserve_keys:
                if isinstance(child, ConfigTree):
                    child._reset_layer(layer, preserve_keys, prefix + [key])
                else:
                    child.reset_layer(layer)
                    if child.is_empty():
                        deletable.append(key)
        for key in deletable:
            del self._children[key]

    def drop_layer(self, layer):
        """Removes the named layer and the value associated with it from the node.

        Parameters
        ----------
        layer : str
            Name of the layer to drop.

        Raises
        ------
        TypeError
            If the node is frozen
        KeyError
            If the named layer does not exist
        """
        if self._frozen:
            raise TypeError('Frozen ConfigTree does not support modification')
        for child in self._children.values():
            child.drop_layer(layer)
        self._layers.remove(layer)

    def unused_keys(self):
        """Lists all keys which are present in the ConfigTree but which have not been accessed."""
        unused = set()
        for k, c in self._children.items():
            if isinstance(c, ConfigNode):
                if not c.has_been_accessed():
                    unused.add(k)
            else:
                for ck in c.unused_keys():
                    unused.add(k+'.'+ck)
        return unused

    def __len__(self):
        return len(self._children)

    def __dir__(self):
        return list(self._children.keys()) + dir(super(ConfigTree, self))

    def __repr__(self):
        return 'ConfigTree(children={}, frozen={})'.format(
            ' '.join([repr(c) for c in self._children.values()]), self._frozen)

    def __str__(self):
        return '\n'.join(['{}:\n    {}'.format(name, str(c).replace('\n', '\n    '))
                          for name, c in self._children.items()])
