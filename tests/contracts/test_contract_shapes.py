"""The example fixture compares a saved example with the live payload by shape: a list and an
object must never pass for each other, and equal records with other values must still match."""

from tests.contracts.conftest import _shape


def test_an_empty_list_and_an_empty_object_have_different_shapes():
    assert _shape([]) != _shape({})
    assert _shape({"data": {"workers": []}}) != _shape({"data": {"workers": {}}})


def test_lists_of_the_same_records_with_other_values_have_equal_shapes():
    assert _shape([{"event": "a", "x": 1}, {"y": "b"}]) == _shape([{"event": "a", "x": 2}, {"y": "c"}])
