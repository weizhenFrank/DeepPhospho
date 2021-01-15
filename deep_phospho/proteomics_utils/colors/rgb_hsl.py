import numpy as np


def rgb_to_hsl(x, h_type='degree'):
    norm_rgb = np.array(x) / 255
    min_value = norm_rgb.min()
    max_value = norm_rgb.max()
    delta_value = max_value - min_value
    l = (min_value + max_value) / 2
    if delta_value == 0:
        h = 0
        s = 0
    else:
        if l < 0.5:
            s = delta_value / (min_value + max_value)
        else:
            s = delta_value / (2 - min_value - max_value)
        delta_rgb = (((max_value - norm_rgb) / 6) + (delta_value / 2)) / delta_value

        for i, one_rgb in enumerate(norm_rgb):
            if one_rgb == max_value:
                sec_calc_index = i + 1 if i < 2 else 0
                h = i / 3 + delta_rgb[i - 1] - delta_rgb[sec_calc_index]
                break
        try:
            if h < 0:
                h += 1
            elif h > 1:
                h -= 1
        except NameError:
            raise
    if h_type == 'degree':
        return h * 360, s, l
    elif h_type == 'decimal':
        return h, s, l
    else:
        raise


def hsl_to_rgb(hsl, h_type='degree'):
    h, s, l = hsl
    if h_type == 'degree':
        h /= 360
    if hsl[1] == 0:
        rgb = [_ * 255 for _ in hsl]
    else:
        if l < 0.5:
            value_2 = l * (s + 1)
        else:
            value_2 = (s + l) - s * l
        value_1 = 2 * l - value_2
        rgb = [255 * hue_to_rgb(value_1, value_2, h + _ / 3) for _ in range(1, -2, -1)]
    for i, one_rgb in enumerate(rgb):
        if one_rgb % 1 > 0.999:
            rgb[i] += 1
    return [int(_) for _ in rgb]


def hue_to_rgb(value_1, value_2, hue):
    if hue < 0:
        hue += 1
    elif hue > 1:
        hue -= 1
    if hue < 1 / 6:
        return value_1 + (value_2 - value_1) * 6 * hue
    elif hue < 1 / 2:
        return value_2
    elif hue < 2 / 3:
        return value_1 + (value_2 - value_1) * (2 / 3 - hue) * 6
    else:
        return value_1


def gradient_hsl(start, end, number, input_color_type='rgb', output_color_type='rgb', value_scale=None):
    if number < 2:
        raise
    if input_color_type.lower() != 'rgb':
        raise

    colors = np.array([
        np.linspace(*each_channel, number) for each_channel in zip(
            rgb_to_hsl(start), rgb_to_hsl(end))]
    ).transpose()

    if output_color_type.lower() == 'rgb':
        colors = [hsl_to_rgb(_) for _ in colors]
    if value_scale:
        colors = [[__ / value_scale for __ in _] for _ in colors]
    return colors


def gradient_rgb(start, end, number, input_color_type='rgb', output_color_type='rgb', value_scale=None):
    if number < 2:
        raise
    if input_color_type.lower() != 'rgb':
        raise

    colors = np.array([
        np.linspace(*each_channel, number) for each_channel in zip(start, end)]
    ).transpose()
    colors = colors.astype(int)
    if value_scale:
        colors = [[__ / value_scale for __ in _] for _ in colors]
    return colors
