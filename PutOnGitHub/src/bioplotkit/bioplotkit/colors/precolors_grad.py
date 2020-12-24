from . import rgb_hsl

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class PreColor:
    red_gradient = np.array(rgb_hsl.gradient_hsl((255, 218, 185), (35, 0, 0), 200)) / 256
    green_gradient = np.array(rgb_hsl.gradient_hsl((240, 255, 240), (0, 35, 0), 200)) / 256
    blue_gradient = np.array(rgb_hsl.gradient_hsl((240, 255, 255), (0, 0, 35), 200)) / 256

    yellow_gradient = np.array(rgb_hsl.gradient_hsl((255, 250, 205), (255, 255, 0), 200)) / 256
    violet_gradient = np.array(rgb_hsl.gradient_hsl((230, 230, 250), (255, 0, 255), 200)) / 256
    cyan_gradient = np.array(rgb_hsl.gradient_hsl((230, 230, 250), (0, 255, 255), 200)) / 256
    purple_gradient = sns.light_palette(np.array((50, 0, 60))/256, input="rgb", n_colors=200)

    rose_gradient = np.array(rgb_hsl.gradient_hsl((250, 230, 230), (255, 1, 1), 200)) / 256
    grass_gradient = np.array(rgb_hsl.gradient_hsl((230, 250, 230), (0, 255, 0), 200)) / 256
    blue_violet_gradient = np.array(rgb_hsl.gradient_hsl((230, 230, 250), (0, 0, 255), 200)) / 256

    grey_gradient = np.array(rgb_hsl.gradient_rgb((255, 255, 255), (0, 0, 0), 200)) / 256


PreColorDict = {
    'Red': PreColor.red_gradient,
    'Green': PreColor.green_gradient,
    'Blue': PreColor.blue_gradient,
    'Yellow': PreColor.yellow_gradient,
    'Violet': PreColor.violet_gradient,
    'Purple': PreColor.purple_gradient,
    'Cyan': PreColor.cyan_gradient,
    'Rose': PreColor.rose_gradient,
    'Grass': PreColor.grass_gradient,
    'Blue-violet': PreColor.blue_violet_gradient,
    'Grey': PreColor.grey_gradient,
}


class FavorGradColor:
    GreyBlueRed = rgb_hsl.gradient_hsl(
        (230, 235, 240), (30, 40, 60), 150, value_scale=256)[::-1] + rgb_hsl.gradient_hsl(
        (245, 250, 255), (230, 235, 240), 35, value_scale=256)[::-1] + rgb_hsl.gradient_hsl(
        (255, 250, 245), (240, 230, 220), 35, value_scale=256) + rgb_hsl.gradient_hsl(
        (240, 230, 220), (100, 10, 10), 150, value_scale=256)
    BlueRed = sns.color_palette('RdBu_r', n_colors=400)
    DardRed = sns.mpl_palette("Reds_d", 400)[::-1]


def draw_grad_precolors():
    c_num = len(PreColorDict)
    f, axes = plt.subplots(1, c_num, figsize=(c_num, 1), dpi=150)

    for ax_index, (c_name, color) in enumerate(PreColorDict.items()):
        ax = axes[ax_index]
        ax.grid(False)

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        for _ in ['bottom', 'top', 'left', 'right']:
            ax.spines[_].set_visible(False)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        n = 200
        _color = color[:n]

        for i, c in enumerate(_color):
            ax.plot((0, 1), (i / n, i / n), color=c)

        ax.annotate(0, (1, 0),
                    textcoords='offset points', xytext=(0.5, 0),
                    va='center', ha='left', fontsize=9, rotation=0)
        ax.annotate(1, (1, 1),
                    textcoords='offset points', xytext=(0, 0),
                    va='center', ha='left', fontsize=9, rotation=0)

        ax.annotate(c_name, (0.5, 1),
                    textcoords='offset points', xytext=(0, 5),
                    va='bottom', ha='center', fontsize=9, rotation=0)
