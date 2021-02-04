
def normalize_intensity(inten, max_num=None):
    """
    The max_num here is the maximum number of intensity, and it will be 1 if is not defined
    """
    if not max_num:
        max_num = 1
    if isinstance(inten, dict):
        max_intensity = max(inten.values())
        scale = max_num / max_intensity
        for each_fragment in list(inten.keys()):
            inten[each_fragment] *= scale
        return inten
    elif isinstance(inten, list):
        max_intensity = max(inten)
        scale = max_num / max_intensity
        return [_ * scale for _ in inten]
    else:
        raise
