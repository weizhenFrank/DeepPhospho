import numpy as np
from scipy.stats import pearsonr
from mskit.metric import similarity

"""
Use all label / Use shared
From array / From dict
array 或 dict 的上一级，prec 水平
返回值包含所有 prec union 的数量，shared 数量，即两个 n
删除完全没有匹配或给一个默认值 0？
"""

"""
1. One side: 一边为 benchmark，另一边缺失补零 or other number
2. Two side: 两边并集，缺失补零
3. Shared: 两边共有
"""


def inten_pair_from_dict(inten1: dict, inten2: dict, method='OneSide', fill_value=0):
    """
    # Get the two intensity lists from input inten dicts and the values are remained through the keys in main_dict
    # The missing values in test_dict will be filled with zero as default
    Input dicts with key-value pairs as [frag_name, intensity]
    """
    inten1_list, inten2_list = [], []
    if method == 'Shared':
        frags = inten1.keys() & inten2.keys()
    elif method == 'OneSide':
        frags = inten1.keys()
    elif method == 'TwoSide':
        frags = inten1.keys() | inten2.keys()
    else:
        raise ValueError(f'The get_frag_list function requires a method param with through_main or shared. {method} is passed now.')
    for frag in frags:
        inten1_list.append(inten1.get(frag, fill_value))
        inten2_list.append(inten2.get(frag, fill_value))
    return inten1_list, inten2_list


def inten_pair_from_array(main_array: np.ndarray, test_array: np.ndarray, filter_func=None):
    """
    Get two intensity pair lists from input two inten array and the remained values are followed
    with the idx of values within the range of defined min_num and max_num in main_array
    Example of filter_func:
        filter_func=lambda x: (x > 0) & (x <= 1)
    """
    if main_array.shape != test_array.shape:
        return None
    main_array = main_array.reshape(-1)
    test_array = test_array.reshape(-1)
    if filter_func:
        used_idx = np.where(filter_func(main_array))
    else:
        used_idx = np.where(main_array)
    return main_array[used_idx], test_array[used_idx]


def get_frag_list(fragment_data1, fragment_data2, method='OneSide'):
    matched_pair_dict = dict()
    shared_prec = fragment_data1.keys() & fragment_data2.keys()
    for prec in shared_prec:
        inten_data1 = fragment_data1[prec]
        inten_data2 = fragment_data2[prec]
        data_list1, data_list2 = inten_pair_from_dict(inten_data1, inten_data2, method=method, fill_value=0)
        matched_pair_dict[prec] = (data_list1, data_list2)
    return matched_pair_dict


def calc_pcc(data1, data2, keep_pvalue=False):
    pcc = pearsonr(data1, data2)
    if keep_pvalue:
        return pcc
    else:
        return pcc[0]


def calc_sa(data1, data2):
    data1 = np.array(data1)
    data2 = np.array(data2)
    norm_data1 = data1 / np.sqrt(sum(np.square(data1)))
    norm_data2 = data2 / np.sqrt(sum(np.square(data2)))
    sa = 1 - 2 * (np.arccos(sum(norm_data1 * norm_data2))) / np.pi
    return sa


def calc_pcc_with_dictdata(data1, data2, method='OneSide', fill_value=0, na_value=None):
    pcc = similarity.pcc(*inten_pair_from_dict(data1, data2, method=method, fill_value=fill_value))
    if na_value is not None and np.isnan(pcc):
        pcc = na_value
    return pcc


def calc_sa_with_dictdata(data1, data2, method='OneSide'):
    return similarity.sa(*inten_pair_from_dict(data1, data2, method=method, fill_value=0))


def frag_pcc(main_frag_data, test_frag_data, min_pairs=3, keep_type='through_main', keep_pvalue=False):
    pcc_dict = dict()
    matched_pair_dict = get_frag_list(main_frag_data, test_frag_data, method=keep_type)
    for prec, (main_list, test_list) in matched_pair_dict.items():
        if len(main_list) < min_pairs or len(test_list) < min_pairs:
            continue
        pcc = calc_pcc(main_list, test_list, keep_pvalue=keep_pvalue)
        if np.isnan(np.min(pcc)):
            continue
        pcc_dict[prec] = pcc
    return pcc_dict


def frag_sa(main_frag_data, test_frag_data, min_pairs=3, keep_type='through_main'):
    sa_dict = dict()
    matched_pair_dict = get_frag_list(main_frag_data, test_frag_data, method=keep_type)
    for prec, (main_list, test_list) in matched_pair_dict:
        if len(main_list) < min_pairs or len(test_list) < min_pairs:
            continue
        sa = calc_sa(main_list, test_list)
        sa_dict[prec] = sa
    return sa_dict


def write_pcc_result(pred_file_path, pcc_dict):
    with open(r'{}.fragment.info'.format(pred_file_path), 'w') as f:
        for _, __ in pcc_dict.items():
            f.write('{}\t{}\n'.format(_, __))


def write_sa_result(pred_file_path, pcc_dict):
    with open(r'{}.sa.info'.format(pred_file_path), 'w') as f:
        for _, __ in pcc_dict.items():
            f.write('{}\t{}\n'.format(_, __))
