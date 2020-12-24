import os


def sum_set_in_list(list_with_sets, return_type='set'):
    summed_list = sum([list(each_set) for each_set in list_with_sets], [])
    if return_type == 'set':
        return set(summed_list)
    elif return_type == 'list':
        return summed_list
    else:
        print('Not supported return type when sum set list')


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


def check_value_len_of_dict(checked_dict, thousands_separator=True):
    for k, v in checked_dict.items():
        v_len = len(v)
        if thousands_separator:
            print(f'{k}: {format(v_len, ",")}')
        else:
            print(f'{k}: {v_len}')


def print_basename_in_dict(path_dict):
    for name, path in path_dict.items():
        print(f'{name}: {os.path.basename(path)}')


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
