import numpy as np
import pytest
import seaborn as sns
import matplotlib.pyplot as plt
import pandas
import os


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


def test_load_tips_dataset():
    tips_df: pandas.DataFrame = sns.load_dataset("tips")

    headers = ['total_bill', 'tip', 'sex', 'smoker', 'day', 'time', 'size']
    assert tips_df.columns.to_list() == headers

    shape = (244, 7)
    assert tips_df.shape == shape

    number_of_elements = 244
    for col in tips_df.count():
        assert col == number_of_elements

    first_line = [16.99, 1.01, 'Female', 'No', 'Sun', 'Dinner', 2]
    assert tips_df.iloc[0].to_list() == first_line

    last_line = [18.78, 3.0, 'Female', 'No', 'Thur', 'Dinner', 2]
    assert tips_df.iloc[-1].to_list() == last_line


def test_set_title_and_labels_of_plot():
    data = [1, 2, 3, 4, 5]
    title = "Title of plot"

    plot: sns.Axes = sns.distplot(data)
    plot.set_title(title)
    assert plot.get_title() == title

    xlabel = "X Axis"
    ylabel = "Y Axis"
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    ax = plt.gca()
    assert ax.get_xlabel() == xlabel
    assert ax.get_ylabel() == ylabel


def test_save_chart_as_png():
    tips_df: pandas.DataFrame = sns.load_dataset("tips")

    g = sns.axisgrid.FacetGrid(tips_df, col="sex")
    g.map(sns.scatterplot, "total_bill", "tip")

    if not os.path.exists("charts"):
        os.mkdir("charts")

    assert os.path.exists("charts")

    g.savefig("charts/siema.png")

    assert os.path.exists("charts/siema.png")


def test_putting_data_to_facegrid():
    tips_df: pandas.DataFrame = sns.load_dataset("tips")

    g = sns.FacetGrid(tips_df, row="smoker", col="time", margin_titles=True)
    g.map(sns.regplot, "size", "total_bill", color=".3", fit_reg=False, x_jitter=.1)

    assert len(g.data) == len(tips_df)

    rows = ['Yes', 'No']
    assert g.row_names == rows

    cols = ['Lunch', 'Dinner']
    assert g.col_names == cols

    assert g._margin_titles


def test_set_palette():
    sns.set_palette("husl")
    palette = sns.color_palette()
    assert palette

    for color in palette:
        assert isinstance(color, tuple)
        assert len(color) == 3
        assert all(0 <= c <= 1 for c in color)


def test_create_histogram_plot():
    penguins = sns.load_dataset("penguins")

    xlabel = "flipper_length_mm"
    color = (1.0, 0.0, 0.0, 0.75)
    sns.histplot(data=penguins, x=xlabel, color=color)

    ax = sns.histplot(penguins)
    assert ax is not None

    number_of_bars = 182
    assert len(ax.patches) == number_of_bars

    ylabel = "Count"
    assert ax.get_ylabel() == ylabel
    assert ax.get_xlabel() == xlabel

    assert ax.patches[0].get_facecolor() == color


def test_set_markers_on_false():
    #  This test will return False, because there is no option to set markers on False
    flights_wide = sns.load_dataset('flights').pivot(index='year', columns='month', values='passengers')
    sns.scatterplot(data=flights_wide, markers=False)

    ax = sns.plt.gca()
    markers_setting = ax.get_lines()[0].get_markersize()
    assert markers_setting == 0
