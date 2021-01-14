from matplotlib import pyplot as plt


def bar_anno(anno_text, x_pos, y_height=None, y_bottom=0,
             thousand_separ=True, add_percent=False, denominator=None,
             ha='center', va='center',
             rotation=90, fontsize=8,
             ax=None):
    if ax is None:
        ax = plt.gca()

    if thousand_separ is True:
        showed_anno = format(anno_text, ',')
    elif thousand_separ is False:
        showed_anno = str(anno_text)
    else:
        showed_anno = None

    if not y_height:
        y_height = anno_text

    y_pos = y_height / 2 + y_bottom
    ax.annotate(showed_anno, (x_pos, y_pos),
                ha=ha, va=va, rotation=rotation, fontsize=fontsize, )


def dyna_bar_anno(anno_text, x_pos, y_height=None, y_bottom=0,
                  thousand_separ=True,
                  ha='center', va='center',
                  rotation=90, fontsize=8,
                  is_dyna_func=lambda x, y: abs(x) < (abs(y) / 2),
                  dyna_pos_func=lambda x, y: 2 * x,
                  auto_ha_va=False,
                  new_ha=None, new_va=None, new_rotation=None,
                  ax=None):
    """
    如果是本身就很低的一个bar但是bottom是0还是会被挡，太菜了，把 dyna func 写成 lambda x, y: x < 500 之类的海星

    TODO 把 dyna func 这部分拿出来单独处理，放进 unit test中
    TODO 这里同时定义两个 func 也太麻烦了
    """

    if not y_height:
        y_height = anno_text

    is_dyna = is_dyna_func(y_height, y_bottom)

    if is_dyna:
        y_height = dyna_pos_func(y_height, y_bottom)

    if auto_ha_va is True and is_dyna:
        va = 'bottom' if y_height > 0 else 'top'
    elif auto_ha_va is False:
        if new_ha and is_dyna:
            ha = new_ha
        if new_va and is_dyna:
            va = new_va
        if new_rotation and is_dyna:
            rotation = new_rotation
    else:
        pass

    bar_anno(anno_text, x_pos, y_height=y_height, y_bottom=y_bottom,
             thousand_separ=thousand_separ,
             ha=ha, va=va,
             rotation=rotation, fontsize=fontsize,
             ax=ax)
