"""
============================
Population Utility Functions
============================

"""
import re


def extract_columns_from_query(query: str) -> set[str]:
    """Extract column names required by a query string."""

    # Extract columns with backticks
    columns = re.findall(r"`([^`]*)`", query)

    # Begin dropping known non-columns from query
    # Remove backticked content
    query = re.sub(r"`[^`]*`", "", query)
    # Remove keywords including "in" and "not in"
    query = re.sub(r"\b(and|if|or|True|False|in|not\s+in)\b", "", query, flags=re.IGNORECASE)
    # Remove quoted strings
    query = re.sub(r"'[^']*'|\"[^\"]*\"", "", query)
    # Remove standalone numbers (not part of identifiers)
    query = re.sub(r"\b\d+\b", "", query)
    # Remove @ references
    query = re.sub(r"@\S+", "", query)
    # Remove list/array syntax
    query = re.sub(r"\[[^\]]*\]", "", query)
    # Remove operators and punctuation but preserve column names
    query = re.sub(r"[!=<>]+|[()&|~\-+*/,.]", " ", query)

    # Combine query words and columns
    query = re.sub(r"\s+", " ", query).strip()
    query_words = [word for word in query.split(" ") if word]
    return set(query_words + columns)


def combine_queries(*queries: str) -> str:
    """Combines any number of queries with an 'and' operator.

    Notes
    -----
    Empty queries (i.e., '') are ignored.
    """
    return " and ".join([f"({query})" for query in filter(None, queries)])
