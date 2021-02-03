import os
import pandas as pd
"""
文件夹中包含导出的谱图，每个文件按precursor命名
"""


def _num_identi(super_sub_script_num):
    if 8320 <= super_sub_script_num <= 8329:
        return 'super'
    else:
        return 'sub'


def _prm_fragment_name(fragment_content):
    fragment_type = fragment_content[0]
    fragment_num = ''
    fragment_charge = '1'
    for _ in fragment_content[1: -1]:
        ord_num = ord(_)
        num_type = _num_identi(ord_num)
        if num_type == 'super':
            fragment_num += str(ord_num - 8320)
        elif num_type == 'sub':
            fragment_charge = str(ord_num - 176)
    fragment_name = f'{fragment_type}{fragment_num}+{fragment_charge}-noloss'
    return fragment_name


def prm_fragment_process(prm_file_content):
    fragment_intensity_dict = dict()
    for each_byte_intensity_line in prm_file_content[3:]:
        each_intensity_line = each_byte_intensity_line.decode('utf-8')
        split_line = each_intensity_line.strip('\n').split('\t')
        if len(split_line) == 4:
            if 'M' in split_line[2]:
                continue
            each_intensity = float(split_line[1])
            each_fragment = split_line[2]
            if ',' not in each_fragment:
                fragment_name = _prm_fragment_name(each_fragment)
                fragment_intensity_dict[fragment_name] = each_intensity
            else:
                multi_fragments = each_fragment.split(', ')
                fragment_name_list = [
                    _prm_fragment_name(_) for _ in multi_fragments]
                single_electro = [_ for _ in fragment_name_list if '+1' in _]
                if single_electro:
                    fragment_name = single_electro[0]
                else:
                    y_fragment_list = [
                        _ for _ in fragment_name_list if 'y' in fragment_name_list]
                    if y_fragment_list:
                        fragment_name = y_fragment_list[0]
                    else:
                        fragment_name = fragment_name_list[0]
                fragment_intensity_dict[fragment_name] = each_intensity
    return fragment_intensity_dict


def read_prm_result(result_folder):
    prm_intensity_dict = dict()
    for each_file in os.listdir(result_folder):
        each_file_path = os.path.join(result_folder, each_file)
        with open(each_file_path, 'rb') as f:
            file_content = f.readlines()
            each_fragment_dict = prm_fragment_process(file_content)
            precursor = os.path.splitext(each_file)[0]
            charge = precursor[-1]
        prec_info_dict = dict()
        prec_info_dict['Fragment'] = each_fragment_dict
        prec_info_dict['Charge'] = charge
        prec_info_dict['Precursor'] = precursor
        prm_intensity_dict[precursor] = prec_info_dict
    return prm_intensity_dict
