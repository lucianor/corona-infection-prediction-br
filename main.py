import os
import time
from datetime import date

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import lmfit

# CSV_FILENAME = "data/time_series_19-covid-Confirmed.csv"
CSV_URL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv"
CSV_URL_DEATHS = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv"
COUNTRY = "United Kingdom"
COUNTRIES = [
    "all",
    "United Kingdom",
    "Austria",
    "Germany",
    "Italy",
    "France",
    "China",
    "Switzerland",
    "US",
]


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def exponential(x, a, k, b):
    return a * np.exp(x * k) + b


def get_data(url):
    print("Downloading new data")
    df = pd.read_csv(url)
    print("Data downloaded")
    return df


def generate_exponential_chart(df, country, title, pathdir):
    # filter just one country
    df = df[df["Country/Region"] == country]
    df = df[df["Province/State"] == country]
    df = df.drop(columns=["Country/Region", "Province/State", "Lat", "Long"])
    df = df.iloc[0]  # convert to pd.Series

    # start with first infections
    df = df[df.values != 0]

    # parse to datetime
    df.index = pd.to_datetime(df.index, format="%m/%d/%y")

    # fit to exponential function
    time_in_days = np.arange(len(df.values))
    poptimal_exponential, pcovariance_exponential = curve_fit(
        exponential, time_in_days, df.values, p0=[0.3, 0.205, 0]
    )

    # Plot current DATA
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(df.index, df.values, "*", label=f"{title} in {country}")
    ax.plot(
        df.index,
        exponential(time_in_days, *poptimal_exponential),
        "g-",
        label="Exponential Fit",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Number of {title}")
    ax.legend()
    ax.grid()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    fig.suptitle(f"{date.today()} - {country} - {title} - Fit")
    fig.autofmt_xdate()
    fig.savefig(
        f"{pathdir}exponential_fit_{country.replace(' ','')}.png", bbox_inches="tight"
    )

    # Compute prediction
    prediction_in_days = 10
    time_in_days = np.arange(
        start=len(df.values), stop=len(df.values) + prediction_in_days
    )
    prediction = exponential(time_in_days, *poptimal_exponential).astype(int)
    df_prediction = pd.Series(prediction)

    # convert index to dates
    df_prediction.index = pd.date_range(
        df.index[-1], periods=prediction_in_days + 1, closed="right"
    )

    df_prediction = df.append(df_prediction)

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df_prediction)

    # Plot prediction
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(df.index, df.values, "*", label=f"{title} in {country}")
    ax.plot(
        df_prediction.index,
        df_prediction.values,
        "r--",
        label=f"Predicted Number of {title}",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Number of {title}")
    ax.legend()
    ax.grid()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    fig.suptitle(f"{date.today()} - {country} - {title} - Extrapolation")
    fig.autofmt_xdate()
    fig.savefig(
        f"{pathdir}exponential_extrapolation_{country.replace(' ','')}.png",
        bbox_inches="tight",
    )


def generate_logistic_chart(df, countries, title, output_path):
    colormap = [
        [0, 0, 0],  # black
        [230, 159, 0],  # orange
        [86, 180, 233],  # sky blue
        [0, 158, 115],  # bluish green
        [240, 228, 66],  # yellow
        [0, 114, 178],  # blue
        [213, 94, 0],  # vermillion
        [204, 121, 167],  # reddish purple
        [255, 255, 255],
    ]
    # colormap = [[c[0],c[1],c[2],0.0] for c in colormap]
    cp = ["#%02x%02x%02x" % (c[0], c[1], c[2]) for c in colormap]
    mpl.rcParams["figure.dpi"] = 300

    fig, ax = plt.subplots(len(countries), 1, sharex="all")
    fig.set_size_inches(10, 2 * len(countries), forward=True)
    for i, country in enumerate(countries):
        if country == "all":
            values = df.sum().values[4:]
        else:
            values = df[df["Country/Region"] == country].sum().values[4:]
        if np.any(np.diff(values > 1)):
            start = np.where(np.diff(values > 1))[0][0]
        else:
            start = 0
        x = np.arange(-len(values) + 1, 1)[start:]
        x2 = np.arange(-len(values) + 1, 30)[start:]

        values = values[start:]

        f_logistic = lambda x, L, k, x0: L / (1 + np.exp(-k * (x - x0)))
        pars = lmfit.Parameters()
        pars.add_many(
            ("L", 5000, True, 0.0, 1e8, None),
            ("k", 0.1, True, 0, 1e5, None),
            ("x0", 0, True, -1e5, 1e5, None),
        )
        gmodel = lmfit.Model(f_logistic)
        out = gmodel.fit(values.astype("float"), x=x, params=pars)
        dely = out.eval_uncertainty(x=x2, sigma=1)
        best = out.eval(x=x2)

        ax[i].plot(x2, best, cp[5])
        ax[i].fill_between(x2, best - dely, best + dely, color=cp[5], alpha=0.25)
        ax[i].plot(x, values, cp[1])

        ax[i].set_title(
            country
            + " - max = "
            + "{:,}".format(out.params["L"].value.astype("int"))
            + " - growth = "
            + "{:0.3}".format(out.params["k"].value)
            + " - mid_point = "
            + "{:.3}".format(out.params["x0"].value)
            + " - R2 = "
            + "{:.3}".format((1 - np.sum(out.residual ** 2.0) / np.sum(values ** 2.0)))
        )

        ax[i].axvline(x=0, color="k", lw=1)
        ax[i].set_ylim([0, np.max(best) * 2.0])

        c_err = out.params["x0"].stderr
        c = out.params["x0"].value
        ax[i].axvline(x=c, color=cp[7])
        ax[i].fill_betweenx(
            ax[i].get_ylim(), c - c_err, c + c_err, color=cp[7], alpha=0.25
        )
        ax[i].set_xlim([np.min(x2), np.max(x2)])
        ax[i].set_ylabel("N cases")
        ax[i].yaxis.set_label_position("right")
    ax[len(ax) - 1].set_xlabel("days from today (negative: past)")
    fig.suptitle(f"{date.today()} - {title}")
    fig.savefig(output_path, bbox_inches="tight")


if __name__ == "__main__":
    EXPONENTIALCONFIRMED = "plots/exponential/confirmedcases/"
    EXPONENTIALDEATHS = "plots/exponential/deaths/"
    LOGISTICALCONFIRMED = "plots/logistical/confirmedcases"
    LOGISTICALDEATHS = "plots/logistical/deaths/"
    mkdir(EXPONENTIALCONFIRMED)
    mkdir(EXPONENTIALDEATHS)
    mkdir(LOGISTICALCONFIRMED)
    mkdir(LOGISTICALDEATHS)
    df = get_data(CSV_URL)
    generate_exponential_chart(df, COUNTRY, "Confirmed cases", EXPONENTIALCONFIRMED)
    generate_logistic_chart(
        df, COUNTRIES, "Confirmed cases", f"{LOGISTICALCONFIRMED}/logistic-plot.png"
    )
    dfd = get_data(CSV_URL_DEATHS)
    generate_exponential_chart(dfd, COUNTRY, "Deaths", EXPONENTIALDEATHS)
    generate_logistic_chart(
        dfd, COUNTRIES, "Deaths", f"{LOGISTICALDEATHS}/logistic-plot.png"
    )
