import pytest

from vivarium.config_tree import ConfigTree


def test_single_layer():
    d = ConfigTree()
    d.test_key = 'test_value'
    d.test_key2 = 'test_value2'

    assert d.test_key == 'test_value'
    assert d.test_key2 == 'test_value2'

    d.test_key2 = 'test_value3'

    assert d.test_key2 == 'test_value3'
    assert d.test_key == 'test_value'


def test_dictionary_style_access():
    d = ConfigTree()
    d.test_key = 'test_value'
    d['test_key2'] = 'test_value2'

    assert d['test_key'] == 'test_value'
    assert d.test_key2 == 'test_value2'


def test_get_missing():
    d = ConfigTree()
    d.test_key = 'test_value'

    # Missing keys should be empty containers
    assert len(d.missing_key) == 0


def test_multiple_layer_get():
    d = ConfigTree(layers=['first', 'second', 'third'])
    d._set_with_metadata('test_key', 'test_value', 'first')
    d._set_with_metadata('test_key', 'test_value2', 'second')
    d._set_with_metadata('test_key', 'test_value3', 'third')

    d._set_with_metadata('test_key2', 'test_value4', 'first')
    d._set_with_metadata('test_key2', 'test_value5', 'second')

    d._set_with_metadata('test_key3', 'test_value6', 'first')

    assert d.test_key == 'test_value3'
    assert d.test_key2 == 'test_value5'
    assert d.test_key3 == 'test_value6'


def test_outer_layer_set():
    d = ConfigTree(layers=['inner', 'outer'])
    d._set_with_metadata('test_key', 'test_value', 'inner')
    d._set_with_metadata('test_key', 'test_value2', 'outer')

    d.test_key = 'test_value3'

    assert d.test_key == 'test_value3'


def test_read_dict():
    d = ConfigTree(layers=['inner', 'outer'])
    d._read_dict({'test_key': 'test_value', 'test_key2': 'test_value2'}, layer='inner')
    d._read_dict({'test_key': 'test_value3'}, layer='outer')

    assert d.test_key == 'test_value3'
    assert d.test_key2 == 'test_value2'


def test_read_dict_nested():
    d = ConfigTree(layers=['inner', 'outer'])
    d._read_dict({'test_container': {'test_key': 'test_value', 'test_key2': 'test_value2'}}, layer='inner')
    d._read_dict({'test_container': {'test_key': 'test_value3'}}, layer='inner')

    assert d.test_container.test_key == 'test_value3'
    assert d.test_container.test_key2 == 'test_value2'

    d._read_dict({'test_container': {'test_key2': 'test_value4'}}, layer='outer')

    assert d.test_container.test_key2 == 'test_value4'


def test_source_metadata():
    d = ConfigTree(layers=['inner', 'outer'])
    d._read_dict({'test_key': 'test_value'}, layer='inner', source='initial_load')
    d._read_dict({'test_key': 'test_value2'}, layer='outer', source='update')

    assert d.metadata('test_key') == [
        {'layer': 'inner', 'default': False, 'source': 'initial_load', 'value': 'test_value'},
        {'layer': 'outer', 'default': True, 'source': 'update', 'value': 'test_value2'}]


def test_exception_on_source_for_missing_key():
    d = ConfigTree(layers=['inner', 'outer'])
    d._read_dict({'test_key': 'test_value'}, layer='inner', source='initial_load')

    with pytest.raises(KeyError) as excinfo:
        d.metadata('missing_key')
    assert 'missing_key' in str(excinfo.value)


def test_drop_layer():
    d = ConfigTree(layers=['a', 'b', 'c'])
    d._set_with_metadata('test_key', 'test_value', 'a')
    d._set_with_metadata('test_key', 'test_value2', 'b')
    d._set_with_metadata('test_key', 'test_value3', 'c')

    assert d.test_key == 'test_value3'
    d.drop_layer('c')
    assert d.test_key == 'test_value2'

    with pytest.raises(ValueError):
        d.drop_layer('c')


def test_reset_layer():
    d = ConfigTree(layers=['a', 'b', 'c'])
    d._set_with_metadata('test_key', 'test_value', 'a')
    d._set_with_metadata('test_key', 'test_value2', 'b')

    assert d.test_key == 'test_value2'
    d.reset_layer('b')
    assert d.test_key == 'test_value'
    d._set_with_metadata('test_key', 'test_value3', 'b')


def test_reset_layer_with_preserved_keys():
    d = ConfigTree(layers=['a', 'b', 'c'])
    d._set_with_metadata('test_key', 'test_value', 'a')
    d._set_with_metadata('test_key2', 'test_value2', 'a')
    d._set_with_metadata('test_key3', 'test_value3', 'a')
    d._set_with_metadata('test_key4', 'test_value4', 'a')
    d._set_with_metadata('test_key', 'test_value5', 'b')
    d._set_with_metadata('test_key2', 'test_value6', 'b')
    d._set_with_metadata('test_key3', 'test_value7', 'b')
    d._set_with_metadata('test_key4', 'test_value8', 'b')

    d.reset_layer('b', preserve_keys=['test_key2', 'test_key3'])
    assert d.test_key == 'test_value'
    assert d.test_key2 == 'test_value6'
    assert d.test_key3 == 'test_value7'
    assert d.test_key4 == 'test_value4'


def test_reset_layer_with_preserved_keys_at_depth():
    d = ConfigTree(layers=['a', 'b', 'c'])
    d._read_dict({
        'test_key': {
            'test_key2': 'test_value',
            'test_key3': {'test_key4': 'test_value2'}
        },
        'test_key5': {
            'test_key6': 'test_value3',
            'test_key7': 'test_value4'}
    }, layer='a')
    d._read_dict({
        'test_key': {
            'test_key2': 'test_value5',
            'test_key3': {'test_key4': 'test_value6'}
        },
        'test_key5': {
            'test_key6': 'test_value7',
            'test_key7': 'test_value8',
        }
    }, layer='b')

    d.reset_layer('b', preserve_keys=['test_key.test_key3', 'test_key5.test_key6'])
    assert d.test_key.test_key2 == 'test_value'
    assert d.test_key.test_key3.test_key4 == 'test_value6'
    assert d.test_key5.test_key6 == 'test_value7'
    assert d.test_key5.test_key7 == 'test_value4'


def test_unused_keys():
    d = ConfigTree({'test_key': {'test_key2': 'test_value', 'test_key3': 'test_value2'}})

    assert d.unused_keys() == {'test_key.test_key2', 'test_key.test_key3'}

    _ = d.test_key.test_key2

    assert d.unused_keys() == {'test_key.test_key3'}

    _ = d.test_key.test_key3

    assert not d.unused_keys()
