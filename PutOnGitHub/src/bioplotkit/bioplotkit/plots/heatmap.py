"""
1. Heatmap with color blocks (with or without edge lines)
2. With color points not blocks
3. Gap with defined rows or cols
"""

from bioplotkit.drawing_area import remove_xy_ticks
from bioplotkit.colors.precolors_grad import FavorGradColor

from matplotlib import pyplot as plt
import numpy as np


def scatter_heatmap(df, use_row=None, use_col=None,
                    row_name_shift=-3.5, col_name_shift=(-2.5, 3.5), col_name_rotation=45, label_fontsize=8,
                    scatter_thres=None, scatter_func=None, scatter_size=25,
                    anno_fontsize=7, anno_x_shift=0, anno_y_shift=-7.5, anno_format='.2f', anno_style='italic',
                    color_bar=None,
                    grid_color='grey', grid_linewidth=0.5, grid_linestyle='--',
                    ax=None, save=None):
    """
    :param df: The dataframe for ploting. If the df contains some rows or cols not desirable to appear in the figure, use use_row or use_col to control it.
    :param use_row: This can either be used to control the prefered rows to plot or change the order of rows.
    :param use_col: Same as use_row.
    :param row_name_shift: The long should the row name (index of df) shifted from the left axis.
    :param col_name_shift: Similar to row_name_shift but this is a tuple to control the x shift and y shift respectively.
    :param col_name_rotation: The angle to rotate the col name.
    :param label_fontsize: Font size of row names and col names (different from the anno_fontsize below).
    :param scatter_thres: The threshold to control the boundary of data points. This can be None, float or a tuple of two float.
    When this is set to None, all values in df will be retained.
    When this is set to a float, the lower threshold is determined and values below this will be dropped before plotting (set to na actually).
    When this is set to a tuple of two floats, the two boundaries are determined and values beyond this range will be dropped (set to na).
    This will also decide how to scale the values to find a correct color from the given color bar.
    :param scatter_func: A function with a point data input, and output None or False if the data should be dropped.
    :param scatter_size: The area size of the scatter plot.

    :return
    """
    if ax is None:
        ax = plt.gca()
    remove_xy_ticks(ax)
    if not color_bar:
        color_bar = FavorGradColor.DardRed

    if use_row:
        df = df.loc[use_row]
    if use_col:
        df = df.loc[: use_col]

    row_num = df.shape[0]
    col_num = df.shape[1]
    color_num = len(color_bar)

    ax.set_xlim(0, col_num)
    ax.set_ylim(0, row_num)

    scaled_df = df.copy()
    if scatter_thres:
        if len(scatter_thres) == 1:
            scaled_df[scaled_df < scatter_thres] = np.nan
            scaled_df -= scatter_thres
            scaled_df /= scaled_df.values.max()
        elif len(scatter_thres) == 2:
            scaled_df[(scaled_df < scatter_thres[0]) | (scaled_df > scatter_thres[1])] = np.nan
            scaled_df -= scatter_thres[0]
            scaled_df /= (scatter_thres[1] - scatter_thres[0])
    else:
        scaled_df -= scaled_df.values.min()
        scaled_df /= scaled_df.values.max()

    for row_index, (row_name, row_data) in enumerate(df[::-1].iterrows()):
        ax.annotate(row_name, (0, row_index + 0.5),
                    textcoords='offset points', xytext=(row_name_shift, 0),
                    va='center', ha='right', fontsize=label_fontsize, rotation=0)
        for col_index, (col_name, data) in enumerate(row_data.iteritems()):
            if scatter_func:
                if not scatter_func(data):
                    continue
            scaled_data = scaled_df.loc[row_name, col_name]
            if np.isnan(scaled_data):
                continue
            try:
                c = np.array(color_bar[int(scaled_data * color_num)]).reshape(1, 3)
            except IndexError:
                c = np.array(color_bar[int(scaled_data * color_num) - 1]).reshape(1, 3)

            x = col_index + 0.5
            y = row_index + 0.5
            ax.scatter(x, y, c=c, s=scatter_size)
            ax.annotate(format(data, anno_format), (x, y),
                        textcoords='offset points', xytext=(anno_x_shift, anno_y_shift), style=anno_style,
                        va='center', ha='center', fontsize=anno_fontsize, rotation=0)

    for col_index, col_name in enumerate(df.columns):
        ax.annotate(col_name, (col_index + 0.5, row_num),
                    textcoords='offset points', xytext=col_name_shift,
                    va='bottom', ha='left', fontsize=label_fontsize, rotation=col_name_rotation)

    for i in ['xy', 'ij']:
        ax.plot(*np.meshgrid(np.arange(0, col_num + 1), np.arange(0, row_num + 1), indexing=i), c=grid_color, linestyle=grid_linestyle, lw=grid_linewidth)

    if save:
        plt.savefig(save + '.PCC.png')

    return df, scaled_df


def multilayer_heatmap():
    """
    Draw multi row in one block with multi df input with customed bar gap and group gap.
    Fig3
    """
    pass
