import yaml

class ConfigNode:
    def __init__(self, layers=None):
        if not layers:
            self._layers = ['base']
        else:
            self._layers = layers
        self._values = {}
        self._frozen = False

    def freeze(self):
        self._frozen = True

    def get_value_with_source(self, layer=None):
        if layer:
            return self._values[layer]
        for layer in reversed(self._layers):
            if layer in self._values:
                return self._values[layer]
        raise KeyError(layer)

    def get_value(self, layer=None):
        return self.get_value_with_source(layer)[1]

    def source(self):
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
        if self._frozen:
            raise TypeError('Frozen ConfigNode does not support assignment')

        if not layer:
            layer = self._layers[-1]
        self._values[layer] = (source, value)

class ConfigTree:
    def __init__(self, data=None, layers=None):
        if not layers:
            self.__dict__['_layers'] = ['base']
        else:
            self.__dict__['_layers'] = layers
        self.__dict__['_children'] = {}
        self.__dict__['_frozen'] = False

        if data:
            self.read_dict(data, layer=self._layers[0], source='initial data')

    def freeze(self):
        self.__dict__['_frozen'] = True
        for child in self._children.values():
            child.freeze()

    def __setattr__(self, name, value):
        self.set_with_metadata(name, value, layer=None, source=None)

    def __getattr__(self, name):
        return self.get_from_layer(name)

    def __contains__(self, name):
        return name in self._children

    def get_from_layer(self, name, layer=None):
        if name not in self._children:
            if self._frozen:
                raise KeyError(name)
            self._children[name] = ConfigTree(layers=self._layers)
        child = self._children[name]
        if isinstance(child, ConfigNode):
            return child.get_value(layer)
        else:
            return child

    def set_with_metadata(self, name, value, layer=None, source=None):
        if self._frozen:
            raise TypeError('Frozen ConfigTree does not support assignment')

        if isinstance(value, dict):
            if name not in self._children or not isinstance(self._children[name], ConfigTree):
                self._children[name] = ConfigTree(layers=self._layers)
            self._children[name].read_dict(value, layer, source)
        else:
            if name not in self._children or not isinstance(self._children[name], ConfigNode):
                self._children[name] = ConfigNode(self._layers)
            child = self._children[name]
            child.set_value(value, layer, source)

    def read_dict(self, data_dict, layer=None, source=None):
        for k, v in data_dict.items():
            self.set_with_metadata(k, v, layer, source)

    def loads(self, data_string, layer=None, source=None):
        data_dict = yaml.load(data_string)
        self.read_dict(data_dict, layer, source)

    def load(self, f, layer=None, source=None):
        if hasattr(f, 'read'):
            self.loads(f.read(), layer=layer, source=source)
        else:
            with open(f) as f:
                self.loads(f.read(), layer=layer, source=source)

    def source(self, key=None):
       if key in self._children:
           return self._children[key].source()
       else:
           head, _, tail = key.partition('.')
           if head in self._children:
               return self._children[head].source(source=tail)
           else:
               raise KeyError(key)

    def __len__(self):
        return len(self._children)
