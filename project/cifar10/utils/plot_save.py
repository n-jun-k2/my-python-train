#!python3.8

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


def scatter_plot(x_dataframe: pd.DataFrame, y_dataframe: pd.Series, title: str):
    """
    散布図を作成し保存する。
    保存形式はpng
    """
    plt.clf()
    temp_df = pd.DataFrame(data=x_dataframe.loc[:, 0:1], index=x_dataframe.index)
    temp_df = pd.concat((temp_df, y_dataframe), axis=1, join="inner")
    temp_df.columns = ["First Vector", "Second Vector", "Label"]
    seaborn_plot = sns.lmplot(x=temp_df.columns[0], y=temp_df.columns[1], hue=temp_df.columns[2], data=temp_df, fit_reg=False)

    ax = plt.gca()
    ax.set_title("Separation of Observations usin " + title)

    seaborn_plot.savefig(f'{title}.png')