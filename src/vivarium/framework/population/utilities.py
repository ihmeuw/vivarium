"""
============================
Population Utility Functions
============================

"""


def combine_queries(*queries: str) -> str:
    """Combines any number of queries with an 'and' operator."""
    return " and ".join([f"({query})" for query in filter(None, queries)])
