"""
============================
Population Utility Functions
============================

"""


def combine_queries(query1: str | list[str], query2: str | list[str]) -> str:
    """Combines two queries with an 'and' operator."""
    query1 = [query1] if isinstance(query1, str) else query1
    query2 = [query2] if isinstance(query2, str) else query2
    return " and ".join(filter(None, query1 + query2))
