import numpy as np
import scipy.stats


def _get_share_fragment_list(main_fragment_data, test_fragment_data):
    prec_for_test = set(
        main_fragment_data.keys()) & set(
        test_fragment_data.keys())
    for test_prec in prec_for_test:
        main_fragment = main_fragment_data[test_prec]
        test_fragment = test_fragment_data[test_prec]
        fragment_for_test = set(
            main_fragment.keys()) & set(
            test_fragment.keys())
        if not fragment_for_test:
            continue
        temp_main_list = []
        temp_test_list = []
        for each_test_fragment in fragment_for_test:
            temp_main_list.append(main_fragment[each_test_fragment])
            temp_test_list.append(test_fragment[each_test_fragment])
        yield temp_main_list, temp_test_list, test_prec


def frag_pcc(main_fragment_data, test_fragment_data, min_pairs=3):
    pcc_dict = dict()
    for temp_main_list, temp_test_list, test_prec in _get_share_fragment_list(
            main_fragment_data, test_fragment_data):
        pcc = scipy.stats.pearsonr(temp_main_list, temp_test_list)
        if np.isnan(np.min(pcc)):
            continue
        pcc_dict[test_prec] = pcc[0]
    return pcc_dict


def frag_sa(main_fragment_data, test_fragment_data):
    sa_dict = dict()
    for temp_main_list, temp_test_list, test_prec in _get_share_fragment_list(
            main_fragment_data, test_fragment_data):
        main_array = np.array(temp_main_list)
        test_array = np.array(temp_test_list)
        normalized_main_array = main_array / \
            np.sqrt(sum(np.square(main_array)))
        normalized_test_array = test_array / \
            np.sqrt(sum(np.square(test_array)))
        sa = 1 - 2 * \
            (np.arccos(sum(normalized_main_array * normalized_test_array))) / np.pi
        if np.isnan(np.min(sa)):
            continue
        sa_dict[test_prec] = sa
    return sa_dict


def func_write_pcc_result(pred_file_path, pcc_dict):
    with open(r'{}.fragment.info'.format(pred_file_path), 'w') as f:
        for _, __ in pcc_dict.items():
            f.write('{}\t{}\n'.format(_, __))


def func_write_sa_result(pred_file_path, pcc_dict):
    with open(r'{}.sa.info'.format(pred_file_path), 'w') as f:
        for _, __ in pcc_dict.items():
            f.write('{}\t{}\n'.format(_, __))
