import pytest
import seaborn as sns
import matplotlib
import pandas
import subprocess


def test_library_installation():
    command = "pip install seaborn"
    success = ["Installing collected packages: seaborn", "Requirement already satisfied: seaborn"]

    output = subprocess.run(command, shell=True, capture_output=True, text=True)
    assert any(message in output.stdout for message in success)


def test_library_uninstallation():
    command = "pip uninstall seaborn -y"
    success = ["Successfully uninstalled", "Skipping seaborn as it is not installed"]

    output = subprocess.run(command, shell=True, capture_output=True, text=True)
    assert any(message in output.stdout for message in success)


def test_load_diamonds_dataset():
    diamonds_df: pandas.DataFrame = sns.load_dataset("diamonds")

    headers = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price', 'x', 'y', 'z']
    assert diamonds_df.columns.to_list() == headers

    shape = (53940, 10)
    assert diamonds_df.shape == shape

    number_of_elements = 53940
    for col in diamonds_df.count():
        assert col == number_of_elements

    first_line = [0.23, "Ideal", "E", "SI2", 61.5, 55., 326, 3.95, 3.98, 2.43]
    assert diamonds_df.iloc[0].to_list() == first_line

    last_line = [0.75, "Ideal", "D", "SI2", 62.2, 55., 2757, 5.83, 5.87, 3.64]
    assert diamonds_df.iloc[-1].to_list() == last_line
