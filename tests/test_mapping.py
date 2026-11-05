"""Round-trip tests for the letter-to-symbol mapping and its prompt example.

Run: python tests/test_mapping.py   (or pytest, if installed)
"""

import re

from stego_inspect.algorithms.mapping import (
    LETTER_TO_SYMBOL_MAPPING,
    MAPPING_EXAMPLE_EXPRESSION,
    decode_letters_to_expression,
    encode_expression_to_letters,
    get_mapping_prompt_section,
)


def test_mapping_is_a_bijection():
    symbols = list(LETTER_TO_SYMBOL_MAPPING.values())
    assert len(symbols) == len(set(symbols))


def test_round_trip_all_symbols():
    expr = "".join(LETTER_TO_SYMBOL_MAPPING.values())
    assert decode_letters_to_expression(encode_expression_to_letters(expr)) == expr


def test_round_trip_digits_and_ops():
    for expr in ["12+34=46", "0123456789", "(7*8)-9/3=53", "1000-1=999"]:
        assert decode_letters_to_expression(encode_expression_to_letters(expr)) == expr


def test_prompt_example_decodes_to_its_expression():
    section = get_mapping_prompt_section()
    m = re.search(r"Example: To encode '([^']+)', use letters: ([A-Z]+)", section)
    assert m, section
    expr, letters = m.groups()
    assert expr == MAPPING_EXAMPLE_EXPRESSION
    assert decode_letters_to_expression(letters) == expr
    assert letters == "ITAHFEFX"
    gloss = section.split(m.group(0), 1)[1].strip().splitlines()[0]
    assert gloss == "I=1, T=2, A=+, H=3, F=4, E==, F=4, X=6"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
