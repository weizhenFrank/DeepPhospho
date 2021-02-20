import os

import numpy as np
import pandas as pd


def sum_set_in_list(list_with_sets, return_type='set'):
    union_set = list_with_sets[0]
    if len(list_with_sets) >= 2:
        for s in list_with_sets[1:]:
            union_set = union_set | s
    if return_type == 'set':
        return union_set
    elif return_type == 'list':
        return list(union_set)
    else:
        print('Not supported return type when sum set list')


def sum_list(nested_list):
    temp = []
    for _ in nested_list:
        temp.extend(_)
    return temp


def align_dict(*dn: dict, columns=None):
    aligned_list = []
    key_union = sorted(set(sum_list(dn)))
    for key in key_union:
        aligned_list.append([di.get(key, np.nan) for di in dn])
    if columns is None:
        columns = [f'D{i}' for i in range(1, len(dn) + 1)]
    return pd.DataFrame(aligned_list, index=key_union, columns=columns)


def split_two_set(set1, set2):
    overlapped = set1 & set2
    set1_unique = set1 - set2
    set2_unique = set2 - set1
    return set1_unique, overlapped, set2_unique


def drop_list_duplicates(initial_list):
    unique_list = list(set(initial_list))
    unique_list = sorted(unique_list, key=initial_list.index)
    return unique_list


def intersect_lists(list_1, list_2, drop_duplicates=True):
    intersected_list = [_ for _ in list_1 if _ in list_2]
    if drop_duplicates:
        return drop_list_duplicates(intersected_list)
    else:
        return intersected_list


def subtract_list(list_1, list_2, drop_duplicates=True):
    subtracted_list = [_ for _ in list_1 if _ not in list_2]
    if drop_duplicates:
        return drop_list_duplicates(subtracted_list)
    else:
        return subtracted_list


def get_coincide_data(dict_1, dict_2):
    shared_keys = list(set(dict_1.keys()) & set(dict_2.keys()))
    value_list_1 = [dict_1[_] for _ in shared_keys]
    value_list_2 = [dict_2[_] for _ in shared_keys]
    return shared_keys, value_list_1, value_list_2


def str_mod_to_list(mod):
    mod_list = [each_mod.split(',') for each_mod in mod.strip(';').split(';')]
    mod_list = [(int(_[0]), _[1]) for _ in mod_list]
    return mod_list


def check_value_len_of_dict(checked_dict: dict, thousands_separator=True, sort_keys=True):
    # TODO sort_keys 可以为 lambda 函数
    if sort_keys:
        keys = sorted(checked_dict.keys())
    else:
        keys = checked_dict.keys()
    for k in keys:
        v = checked_dict[k]
        v_len = len(v)
        if thousands_separator:
            print(f'{k}: {format(v_len, ",")}')
        else:
            print(f'{k}: {v_len}')


class XmlListConfig(list):
    def __init__(self, x_list):
        super(XmlListConfig, self).__init__()
        for element in x_list:
            if element:
                if len(element) == 1 or element[0].tag != element[1].tag:
                    self.append(XmlDictConfig(element))
                elif element[0].tag == element[1].tag:
                    self.append(XmlListConfig(element))
            elif element.text:
                text = element.text.strip()
                if text:
                    self.append(text)


class XmlDictConfig(dict):
    def __init__(self, parent_element):
        super(XmlDictConfig, self).__init__()
        if parent_element.items():
            self.update(dict(parent_element.items()))
        for element in parent_element:
            if element:
                if len(element) == 1 or element[0].tag != element[1].tag:
                    x_dict = XmlDictConfig(element)
                else:
                    x_dict = {element[0].tag: XmlListConfig(element)}
                if element.items():
                    x_dict.update(dict(element.items()))
                self.update({element.tag: x_dict})
            elif element.items():
                self.update({element.tag: dict(element.items())})
            else:
                self.update({element.tag: element.text})


def xml_to_dict(xml_context):
    from xml.etree import cElementTree as ElementTree

    _root = ElementTree.XML(xml_context)
    _xml_dict = XmlDictConfig(_root)
    return _xml_dict
