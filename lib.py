"""
Helper functions for Jupyter notebooks
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns


def fetch_populations():
    """
    Fetches current data about country population
    and returns it as a mapping from countr-name
    to the population value.

    Some keys are duplicated to allow for country-name
    disambiguations which are returned from the API.
    """
    response = requests.get(
        "https://restcountries.eu/rest/v2/all",
        params={"fields": "name;population;altSpellings"},
    )
    output = {}

    for row in response.json():
        if row["name"] in output and output[row["name"]] != row["population"]:
            raise KeyError(
                "%s already in mapping with differing value!" % row["name"]
            )
        output[row["name"]] = row["population"]
        for altname in row["altSpellings"]:
            if altname in output and output[altname] != row["population"]:
                raise KeyError(
                    "%s already in mapping with differing value!" % altname
                )
            output[altname] = row["population"]
    return output


def population_getter(country_pop):
    """
    Generates a pandas-helper to fetch country-population.

    :param country_pop: A dictionary mapping the country name to the
        population number.
    """

    def get_population(row, source_col):
        """
        Helper for a Pandas DataFrame to return the population count.

        :param row: The Pandas record/row
        :param source_col: The name of the column containing the country name
        """
        name = row[source_col]
        if name == "United Kingdom":
            name = "UK"
        return country_pop.get(name, np.NAN)

    return get_population


def to_timeseries(dataframe, var_name):
    """
    Converts the dataframe from the Johns-Hopins data-set
    which stores each date as a column, into a new data-set
    using a new row for each date.
    """
    timeseries = dataframe.melt(
        id_vars=["Province/State", "Country/Region", "Lat", "Long"],
        var_name="Date",
        value_name=var_name,
    )
    timeseries["Date"] = pd.to_datetime(timeseries["Date"], format="%m/%d/%y")
    date_indexed = timeseries.set_index("Date")
    return date_indexed.drop(["Lat", "Long"], axis="columns")


def split(dataframe):
    """
    Splits a data-frame into smaller data-frames, one for each country,
    sorted by date.
    """
    fragments = {}
    for name in dataframe["Country/Region"].unique():
        fragments[name] = dataframe[
            dataframe["Country/Region"] == name
        ].sort_values("Date")
    return fragments


def make_shifer(country_pop):
    """
    Creates a pandas-helper function to shift a dataframe by a number of days.

    :param country_pop: A dictionary mapping country names to population number
    """

    def cases_after_first_n(dataframe, reference_us_population):
        """
        Returns a new data-frame which only contains cases after the firt
        time the "confirmed" counter hit a certain percentage of the
        population. This percentage is calculated based on the US as
        reference. If this value is "500", then this will calculate how much
        of a percentage "500" represents for the population of the US and
        apply the same percentage to the country in the given data-frame.

        This new frame is indexed by the amount of days after the value was
        reached.
        """
        if len(dataframe["Country/Region"].unique()) > 1:
            raise Exception(
                "This function must me called with a data-frame of only one "
                "country"
            )

        reference_rate = reference_us_population / country_pop["US"]
        n = dataframe.iloc[0].population * reference_rate

        matching_rows = dataframe.index[dataframe["confirmed"] > n].tolist()
        if len(matching_rows) == 0:
            return None
        date_of_first_n = matching_rows[0]
        trimmed = dataframe[dataframe.index > date_of_first_n].reset_index()
        trimmed["days_after_cutoff"] = (
            trimmed["Date"] - date_of_first_n
        ).astype("timedelta64[D]")
        return trimmed.set_index("days_after_cutoff")

    return cases_after_first_n


def shift(dataframe, field):
    """
    Create two new fields derived from *field_name*. These new
    fields are added with prefixes in the column names:

    * ``previous_<field>``: The same values as in ``<field>``,
      but offset by one row.
    * ``delta_<field>``: The difference between ``<field>``
      and ``previous_<field>``

    Example:

    >>> dataframe = pd.DataFrame([
    ...    [1, 2, 3],
    ...    [6, 6, 6],
    ...    [7, 7, 7]],
    ...    columns=['a', 'b', 'c'])
    >>> dataframe
       a  b  c
    0  1  2  3
    1  6  6  6
    2  7  7  7
    >>> shift(dataframe, 'b')
    >>> dataframe
       a  b  c  previous_b  delta_b
    0  1  2  3         NaN      NaN
    1  6  6  6         2.0      4.0
    2  7  7  7         6.0      1.0
    """
    dataframe["previous_" + field] = dataframe[field].shift()
    dataframe["delta_" + field] = (
        dataframe[field] - dataframe["previous_" + field]
    )


def shift_all(dataframe):
    """
    Add new columns to a dataframe which are shifted by one row.

    See :py:func:`~.shift` for more details
    """
    shift(dataframe, "confirmed_per_capita")
    shift(dataframe, "recovered_per_capita")
    shift(dataframe, "deaths_per_capita")
    shift(dataframe, "confirmed")
    shift(dataframe, "recovered")
    shift(dataframe, "deaths")


def plot(dataframe, field):
    """
    Plot *field* from *dataframe*
    """
    plt.subplots(figsize=(15, 10))
    sns.set_palette("bright")
    sns.lineplot(
        x=dataframe.index, y=field, data=dataframe, hue="Country/Region"
    )


def smooth(dataframe, field):
    """
    Run a rolling-window average over ``field`` and create
    it as a new field in ``dataframe`` with the prefix ``smooth_``
    """
    dataframe["smooth_" + field] = (
        dataframe[field].rolling(3, center=True).mean()
    )


def prepare_for_plot(dataframe, country_pop):
    countrydata = split(dataframe)
    trimmed_countrydata = {}
    shifter = make_shifer(country_pop)
    for country, frame in countrydata.items():
        trimmed = shifter(frame, 100)
        if trimmed is None:
            continue
        shift_all(trimmed)
        smooth(trimmed, "delta_confirmed_per_capita")
        trimmed_countrydata[country] = trimmed
    output = (
        pd.concat(trimmed_countrydata.values())
        .reset_index()
        .set_index(["days_after_cutoff"])
    )
    return output
