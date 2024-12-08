from __future__ import annotations

import re

from uuid import uuid4

__all__ = [
    "simplify_quotations",
    "generate_id",
]

single_quote_regex = re.compile(r"[‚‘’′‵`‛]")
double_quote_regex = re.compile(r"[„“”″‶″‴〃‷]")

def simplify_quotations(text: str) -> str:
    """
    Simplify the quotation marks in a string - for example, turning
    angled quotes into straight quotes. Applies to both single and
    double quote marks.
    """
    text = single_quote_regex.sub("'", text)
    text = double_quote_regex.sub('"', text)
    return text

def generate_id() -> str:
    """
    Generate a random ID string.
    """
    return str(uuid4())
