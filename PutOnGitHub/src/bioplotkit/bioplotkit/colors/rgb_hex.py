
def hex_to_rgb(hex_color, base=256):
    return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))


def rgb_to_hex(rgb):
    return ''.join([hex(one_channel)[2:] for one_channel in rgb])
