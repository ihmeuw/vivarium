from __future__ import annotations

import pytest

from vivarium.framework.population.utilities import (
    combine_queries,
    extract_columns_from_query,
)


@pytest.mark.parametrize(
    "query, expected",
    [
        ("", set()),
        (
            (
                # Basic
                "alive == 'Alive' and is_aged_out == False and "
                # No spaces
                "answer==42 or "
                "answer_str=='forty-two' or "
                "43!=correct_answer and "
                "duck!=goose or "
                # Mixed operators and casing
                "(10 < age < 20 OR sex == 'Female') IF tiger == 'hobbes' AND "
                # Column-column comparisons and @constants
                "(bar >= baz if @some_const <= 100) or "
                # Names w/ 'and', 'or', or 'if'
                "band == xplor or iffy == 'sketchy' and "
                # Underscores
                "some_col == True or "
                "test_column_1 != 5 and "
                # Casing
                "Foo != Bar and "
                # Special names requiring backticks
                "`spaced column` == False or "
                "`???` != 'unknown' and "
                "`column(1)` < 50 or "
                "`column[2]` < 50 or "
                "`column{3}` < 50 or "
                # Quotes
                "`\"quz\"` == 'value' or "
                'nothing != "something" and '
                # in logic
                "color in ['red', 'blue', 'green'] or "
                "shape not in ['circle', 'square']"
            ),
            {
                "alive",
                "is_aged_out",
                "answer",
                "answer_str",
                "correct_answer",
                "duck",
                "goose",
                "age",
                "sex",
                "tiger",
                "bar",
                "baz",
                "band",
                "xplor",
                "iffy",
                "some_col",
                "test_column_1",
                "Foo",
                "Bar",
                "spaced column",
                "???",
                "column(1)",
                "column[2]",
                "column{3}",
                '"quz"',
                "nothing",
                "color",
                "shape",
            },
        ),
    ],
)
def test_extract_columns_from_query(query: str, expected: set[str]) -> None:
    query_columns = extract_columns_from_query(query)
    assert query_columns == expected


@pytest.mark.parametrize(
    "queries, expected_query",
    [
        [("", ""), ""],
        [("alive == 'alive'", "age < 5"), "(alive == 'alive') and (age < 5)"],
        [
            ("alive == 'alive' or is_aged_out == False", "age < 5", "sex == 'Female'"),
            "(alive == 'alive' or is_aged_out == False) and (age < 5) and (sex == 'Female')",
        ],
    ],
)
def test_combine_queries(queries: tuple[str, ...], expected_query: str) -> None:
    combined = combine_queries(*queries)
    assert combined == expected_query
