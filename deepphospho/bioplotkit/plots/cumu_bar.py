import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bioplotkit.drawing_area import remove_target_spines
from bioplotkit.drawing_area import set_thousand_separate


def get_exp_ratio_idx(data: pd.Series, exp_ratio):
    data = data.cumsum() / data.sum()
    exp_ident = []
    for ratio in exp_ratio:
        ident = data[data > ratio].index[0]
        exp_ident.append(ident)
    return data, exp_ident


def cumu_bar(data, exp_ratio=(0.8, 0.9, 0.95, 0.99),
             max_ylim=5000, xlim=(6, 53),
             left_axis_pos=5.5, right_axis_pos=53.5, bottom_axis_pos=-8,

             mark_min_max=True,
             bar_width=0.6, bar_color='#CDC5BF',
             mark_bar_width=0.6, mark_bar_color='#8B8682', mark_line_style='dashed',
             cumu_line_color='#EE7600',
             vline_color='#CDC9A5',
             xlabel='Peptide length', ylabel='Number of stripped peptide',
             cumu_ylabel='Cumulation percent',
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
    ser = ser.sort_index()

    ax_cumu = ax.twinx()

    remove_target_spines(['top', 'right'], ax=ax)
    remove_target_spines(['top', 'left', 'bottom'], ax=ax_cumu)

    min_data_x, max_data_x = ser.index.min(), ser.index.max()
    min_data_y, max_data_y = ser.min(), ser.max()

    percent_cumu_ser, marked_idx = get_exp_ratio_idx(ser, exp_ratio=exp_ratio)
    cumu_num = percent_cumu_ser * max_data_y

    if max_ylim:
        ...
    else:
        ...

    for idx, num in ser.items():
        if idx in marked_idx:
            ax.bar(idx, num, width=mark_bar_width, bottom=True, color=mark_bar_color)
            ax.vlines(x=idx, ymin=num, ymax=cumu_num.loc[idx],
                      colors=vline_color, linestyles=mark_line_style, label='')
        elif mark_min_max and idx in [min_data_x, max_data_x]:
            ax.bar(idx, num, width=mark_bar_width, bottom=True, color=mark_bar_color)
        else:
            ax.bar(idx, num, width=bar_width, bottom=True, color=bar_color)

    cumu_line = ax_cumu.plot(ser.index.tolist(), cumu_num.values, color=cumu_line_color)

    ax.set_ylim(0, max_ylim)
    ax.set_xlim(*xlim)
    ax.set_xticks([min_data_x, *marked_idx, max_data_x])
    set_thousand_separate(ax=ax, axis='y')

    cumu_ax_yscale = max_data_y
    ax_cumu.set_ylim(bottom=0, top=max_ylim, emit=True, auto=False, ymin=None, ymax=None)
    ax_cumu.set_yticks(np.arange(0, 1.1, 0.1) * cumu_ax_yscale)
    ax_cumu.set_yticklabels(['{}%'.format(_) for _ in range(0, 101, 10)])

    ax.spines['left'].set_position(('data', left_axis_pos))
    ax.spines['bottom'].set_position(('data', bottom_axis_pos))
    ax_cumu.spines['right'].set_position(('data', right_axis_pos))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax_cumu.set_ylabel(cumu_ylabel)

    ax.set_title(title)
