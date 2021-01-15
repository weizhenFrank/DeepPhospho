from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter


def remove_xy_ticks(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


def remove_target_spines(spine_pos, ax=None):
    if ax is None:
        ax = plt.gca()
    if isinstance(spine_pos, str):
        spine_pos = [spine_pos]
    for _ in spine_pos:
        ax.spines[_].set_visible(False)


def remove_all_spines(ax=None):
    if ax is None:
        ax = plt.gca()
    for _ in ['top', 'right', 'bottom', 'left']:
        ax.spines[_].set_visible(False)


def remove_all_bone_info(ax=None):
    if ax is None:
        ax = plt.gca()
    remove_xy_ticks(ax)
    remove_all_spines(ax)


def set_thousand_separate(ax=None, axis=('x', 'y')):
    if ax is None:
        ax = plt.gca()
    if not isinstance(axis, tuple):
        axis = tuple(axis)
    for a in axis:
        if a == 'x':
            ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
        elif a == 'y':
            ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))


def set_bottom_spine_pos0(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.spines['bottom'].set_position(('data', 0))
