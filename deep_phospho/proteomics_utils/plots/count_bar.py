import pandas as pd
import matplotlib.pyplot as plt

from deep_phospho.proteomics_utils.drawing_area import set_thousand_separate


def count_bar(data, sort_index=True,
              ylim=None, xlim=None,
              bar_width=0.6, bar_color=None,
              anno_format=',', anno_rotation=90,
              x_ticks=None, x_tick_rotation=0,
              y_tick_thousand=True,
              xlabel='Missed cleavage', ylabel='Number of peptides',
              title='',
              ax=None):
    if ax is None:
        ax = plt.gca()

    if isinstance(data, list):
        ser = pd.Series(dict(data))
    elif isinstance(data, dict):
        ser = pd.Series(data)
    elif isinstance(data, pd.Series):
        ser = data
    else:
        raise TypeError
    if sort_index:
        ser = ser.sort_index()

    data_keys, data_values = list(zip(*pd.Series(ser).items()))
    if x_ticks is not None:
        bar_xsite = x_ticks
    else:
        bar_xsite = list(range(1, len(data_keys) + 1))

    ax.bar(bar_xsite, data_values, width=bar_width, color=bar_color)
    for site, value in zip(bar_xsite, data_values):
        ax.annotate(format(value, anno_format), (site, value / 2),
                    ha='center', rotation=anno_rotation)

    ax.set_xticks(bar_xsite)
    ax.set_xticklabels(data_keys, rotation=x_tick_rotation)

    if y_tick_thousand:
        set_thousand_separate(ax, 'y')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
