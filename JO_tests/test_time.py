import numpy as np
import pytest
import seaborn as sns
import matplotlib.pyplot as plt
import pandas
import datetime


def test_time_of_putting_data_into_facetgrid():
    diamonds_df: pandas.DataFrame = sns.load_dataset("diamonds")

    start_time = datetime.datetime.now()
    g = sns.axisgrid.FacetGrid(diamonds_df, col="color")
    g.map(sns.scatterplot, 'carat', 'cut', 'clarity', 'depth', 'table', 'price', 'x', 'y', 'z')
    end_time = datetime.datetime.now()

    exe_time = end_time - start_time
    assert exe_time.total_seconds() <= 1
