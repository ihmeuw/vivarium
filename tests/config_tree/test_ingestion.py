from vivarium.config_tree import ConfigTree

TEST_YAML_ONE = '''
test_section:
    test_key: test_value
    test_key2: test_value2
test_section2:
    test_key: test_value3
    test_key2: test_value4
'''


def test_load_yaml_string():
    d = ConfigTree()
    d._loads(TEST_YAML_ONE, source='inline_test')

    assert d.test_section.test_key == 'test_value'
    assert d.test_section.test_key2 == 'test_value2'
    assert d.test_section2.test_key == 'test_value3'


def test_load_yaml_file(tmpdir):
    tmp_file = tmpdir.join('test_file.yaml')
    tmp_file.write(TEST_YAML_ONE)

    d = ConfigTree()
    d._load(str(tmp_file))

    assert d.test_section.test_key == 'test_value'
    assert d.test_section.test_key2 == 'test_value2'
    assert d.test_section2.test_key == 'test_value3'
