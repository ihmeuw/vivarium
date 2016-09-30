import pytest

from ceam.framework.components import apply_defaults

def test_apply_defaults_no_comparisons():
    config = {
            "components": [
                "component_one",
                "component_two"
            ]}
    comparisons = apply_defaults(config)

    assert 'base' in comparisons
    assert set(comparisons['base']['components']) == {'component_one', 'component_two'}

def test_apply_defaults_with_comparison():
    config = {
            "components": [
                "component_one",
                "component_two"
            ],
            "comparisons": [
                {
                    "name": "comparison_one",
                    "components": ["component_three"]
                },
                {
                    "name": "comparison_two",
                    "components": ["component_four"]
                }
            ]}
    comparisons = apply_defaults(config)

    assert 'comparison_one' in comparisons
    assert 'comparison_two' in comparisons
    assert 'base' not in comparisons
    assert set(comparisons['comparison_one']['components']) == {'component_one', 'component_two', 'component_three'}
    assert set(comparisons['comparison_two']['components']) == {'component_one', 'component_two', 'component_four'}
