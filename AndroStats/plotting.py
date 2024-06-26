import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns  # type: ignore
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Any
from AndroStats.performance import BlandAltmanCalculation


def plot_prediction_comparision(df: pd.DataFrame, true_label: str, prediction_label: str) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x=true_label, y=prediction_label, ax=ax)
    ax.set_title(f"{prediction_label} vs {true_label}")
    ax.set_xlabel(true_label)
    ax.set_ylabel(prediction_label)
    return fig


def bland_altman(df: pd.DataFrame, true_label: str, prediction_label: str) -> Figure:
    bac: BlandAltmanCalculation = BlandAltmanCalculation()

    true_values: NDArray[np.float64] = df[true_label].to_numpy()
    prediction_values: NDArray[np.float64] = df[prediction_label].to_numpy()
    bland_altman_data: Dict[str, Any] = bac.calculate(true_values=true_values, predicted_values=prediction_values)

    df["mean"] = bland_altman_data["mean"]
    df["difference"] = bland_altman_data["diff"]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x="mean", y="difference", ax=ax)
    sns.lineplot(x=[0, 100], y=[bland_altman_data["mean_bias"], bland_altman_data["mean_bias"]], color="black", ax=ax)
    sns.lineplot(x=[0, 100], y=[bland_altman_data["upper_limit"], bland_altman_data["upper_limit"]], color="black", linestyle="--", ax=ax)
    sns.lineplot(x=[0, 100], y=[bland_altman_data["lower_limit"], bland_altman_data["lower_limit"]], color="black", linestyle="--", ax=ax)
    ax.set_title(f"{prediction_label} vs {true_label}")
    ax.set_xlabel(true_label)
    ax.set_ylabel(prediction_label)
    return fig
