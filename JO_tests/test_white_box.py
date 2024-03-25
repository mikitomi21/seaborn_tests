import pytest
import pandas as pd
from seaborn._core.data import handle_data_source
from seaborn.utils import load_dataset


@pytest.mark.parametrize("input_data, expected_output_type", [
    (pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}), pd.DataFrame),
    ({'A': [1, 2, 3], 'B': [4, 5, 6]}, dict),
    (None, type(None)),
])
def test_handle_data_source(input_data, expected_output_type):
    result = handle_data_source(input_data)
    assert isinstance(result, expected_output_type)


@pytest.mark.parametrize("invalid_data", [
    123,
    1 / 3,
    "d",
    (1, 2, 3, 4, 5, 6)
])
def test_handle_data_source_invalid_input(invalid_data):
    with pytest.raises(TypeError):
        handle_data_source(invalid_data)


@pytest.mark.parametrize("invalid_data", [
    1 / 3,
    "tips.txt",
    "flights.jpg",
])
def test_load_dataset_value_error(invalid_data):
    with pytest.raises(ValueError):
        load_dataset(invalid_data)


def test_load_dataset_more_helpful_error():
    obj = pd.DataFrame()
    with pytest.raises(TypeError):
        load_dataset(obj)
