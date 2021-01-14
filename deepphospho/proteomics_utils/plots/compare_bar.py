from deepphospho.proteomics_utils.colors import PreColorDict
from deepphospho.proteomics_utils.drawing_area import ax_utls
from deepphospho.proteomics_utils.annotations import bar_anno, dyna_bar_anno

from matplotlib import pyplot as plt


def comp_bar(data_dict, base_key=None, comp_keys=None,
             filter_func=None,  # 对 input data dict 的每个 value 做一步 filter
             base_x_pos=1,
             bar_width=0.4,
             share_color=None, new_color=None, loss_color=None,
             bar_edge_color='grey', bar_edge_width=0.2,
             anno_rotation=90, anno_fontsize=8, anno_ha='center', anno_va='center',
             label_name=None, label_shift=0.1,
             ylabel='', title='',
             ax=None, save=None):

    if ax is None:
        ax = plt.gca()

    if share_color is None:
        share_color = PreColorDict['Blue'][75]
    if new_color is None:
        new_color = PreColorDict['Red'][60]
    if loss_color is None:
        loss_color = PreColorDict['Grey'][60]

    ax_utls.remove_xy_ticks(ax)
    ax_utls.remove_target_spines(('right', 'top'), ax)
    ax_utls.set_bottom_spine_pos0(ax)

    base_data = set(data_dict[base_key])
    if filter_func:
        base_data = filter_func(base_data)
    base_len = len(base_data)
    ax.bar(base_x_pos, base_len, width=bar_width, bottom=0,
           color=share_color, lw=bar_edge_width, edgecolor=bar_edge_color)
    bar_anno(base_len, base_x_pos, y_bottom=0, thousand_separ=True,
             ha=anno_ha, va=anno_va, rotation=anno_rotation, fontsize=anno_fontsize, ax=ax)

    for key_index, each_comp_key in enumerate(comp_keys):
        bar_x_pos = key_index + base_x_pos + 1
        comp_data = set(data_dict[each_comp_key])
        if filter_func:
            comp_data = filter_func(comp_data)

        new_data_num = len(comp_data - base_data)
        share_data_num = len(comp_data & base_data)
        loss_data_num = len(base_data - comp_data)

        # Share
        ax.bar(bar_x_pos, share_data_num, width=bar_width, bottom=0,
               color=share_color, lw=bar_edge_width, edgecolor=bar_edge_color)
        bar_anno(share_data_num, bar_x_pos, y_bottom=0, thousand_separ=True,
                 ha=anno_ha, va=anno_va, rotation=anno_rotation, fontsize=anno_fontsize, ax=ax)

        # New
        ax.bar(bar_x_pos, new_data_num, width=bar_width, bottom=share_data_num,
               color=new_color, lw=bar_edge_width, edgecolor=bar_edge_color)
        dyna_bar_anno(new_data_num, bar_x_pos, y_bottom=share_data_num, thousand_separ=True,
                      ha=anno_ha, va=anno_va, rotation=anno_rotation, fontsize=anno_fontsize,
                      auto_ha_va=True, ax=ax)

        # Loss
        ax.bar(bar_x_pos, -loss_data_num, width=bar_width, bottom=0,
               color=loss_color, lw=bar_edge_width, edgecolor=bar_edge_color)
        dyna_bar_anno(loss_data_num, bar_x_pos, y_height=-loss_data_num, y_bottom=0, thousand_separ=True,
                      ha=anno_ha, va=anno_va, rotation=anno_rotation, fontsize=anno_fontsize,
                      auto_ha_va=True, ax=ax)

    if label_name:
        for label_index, each_name in enumerate(label_name):
            ax.annotate(each_name, (label_index + 1 + bar_width / 2 + label_shift, ax.get_ylim()[0]),
                        rotation=90, va='bottom', ha='left', fontsize=anno_fontsize)

    # ax.set_ylabel(ylabel)  TODO 这里因为删除了 xy ticks 需要单独加一个 text
    ax.set_title(title)

    if save:
        plt.savefig(save + '.Comp.png')
