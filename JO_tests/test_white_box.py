import pytest
import pandas as pd
from seaborn._core.data import handle_data_source
from seaborn._core.properties import Fill


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


@pytest.fixture
def fill_instance():
    return Fill()


@pytest.mark.parametrize("input_data, expected_output", [
    (0, []),
    (1, [True]),
    (2, [True, False]),
    (3, [True, False, True])
])
def test_fill_default_values_valid_input(fill_instance, input_data, expected_output):
    result = fill_instance._default_values(input_data)
    assert result == expected_output


def test_fill_default_values_warning_message(fill_instance):
    with pytest.warns(UserWarning, match="The variable assigned to .* has more than two levels, so .* values will "
                                         "cycle and may be uninterpretable"):
        fill_instance._default_values(3)
