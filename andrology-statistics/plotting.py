import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_prediction_comparision(
    df: pd.DataFrame, true_label: str, prediction_label: str
):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x=true_label, y=prediction_label, ax=ax)
    ax.set_title(f"{prediction_label} vs {true_label}")
    ax.set_xlabel(true_label)
    ax.set_ylabel(prediction_label)
    return fig, ax
