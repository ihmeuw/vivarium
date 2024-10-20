from __future__ import annotations

from datetime import datetime

from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.randomness.index_map import IndexMap
from vivarium.framework.resource.group import ResourceGroup
from vivarium.framework.resource.resource import Column
from vivarium.framework.values import Pipeline, ValueSource


def dummy_producer() -> str:
    return "resources!"


def test_resource_group() -> None:
    r_type = "column"
    r_names = [str(i) for i in range(5)]
    r_producer = dummy_producer
    r_dependencies = [
        Column("an_interesting_column"),
        Pipeline("baz"),
        RandomnessStream("bar", lambda: datetime.now(), 1, IndexMap()),
        ValueSource("foo"),
    ]

    rg = ResourceGroup(r_type, r_names, r_producer, r_dependencies)

    assert rg.type == r_type
    assert rg.names == [f"{r_type}.{name}" for name in r_names]
    assert rg.producer == dummy_producer
    assert rg.dependencies == [
        "column.an_interesting_column",
        "value.baz",
        "stream.bar",
        "value_source.foo",
    ]
    assert list(rg) == rg.names
