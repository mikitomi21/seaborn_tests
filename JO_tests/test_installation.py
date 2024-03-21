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