from datetime import datetime

import pandas as pd
import pytest

from tests.helpers import ColumnCreator
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.randomness.index_map import IndexMap
from vivarium.framework.resource import Resource
from vivarium.framework.resource.resource import Column, NullResource
from vivarium.framework.values import AttributePipeline, Pipeline, ValueModifier, ValueSource


def test_resource_id() -> None:
    resource = Resource("value_source", "test", ColumnCreator())
    assert resource.resource_id == "value_source.test"


@pytest.mark.parametrize(
    "resource, is_initialized",
    [
        (Pipeline("foo"), False),
        (ValueSource(Pipeline("bar"), lambda: 1, ColumnCreator()), False),
        (ValueModifier(Pipeline("baz"), lambda: 1, ColumnCreator()), False),
        (AttributePipeline("foo"), False),
        (
            ValueSource(
                AttributePipeline("bar"), lambda idx: pd.DataFrame(), ColumnCreator()
            ),
            False,
        ),
        (ValueModifier(AttributePipeline("baz"), lambda: 1, ColumnCreator()), False),
        (Column("foo", ColumnCreator()), True),
        (
            RandomnessStream("bar", lambda: datetime.now(), 1, IndexMap(), ColumnCreator()),
            False,
        ),
        (NullResource(0, ColumnCreator()), True),
    ],
)
def test_resource_is_initialized(resource: Resource, is_initialized: bool) -> None:
    assert resource.is_initialized == is_initialized
