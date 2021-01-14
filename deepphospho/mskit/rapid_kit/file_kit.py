import os
import time
import pickle
import pandas as pd


def file_prefix_time(with_dash=False):
    curr_time = time.strftime('%Y%m%d', time.localtime())
    prefix = curr_time + '-' if with_dash else curr_time
    return prefix


def pd_read_csv_skip_row(file, comment=None, **kwargs):
    if os.stat(file).st_size == 0:
        raise ValueError("File is empty")
    with open(file, 'r') as f:
        pos = 0
        cur_line = f.readline()
        while cur_line.startswith(comment):
            pos = f.tell()
            cur_line = f.readline()
            f.seek(pos)
    return pd.read_csv(f, **kwargs)


def read_one_col_file(file, skiprows=None):
    with open(file, 'r') as f:
        one_col_list = [row.strip('\n') for row in f.readlines()]
        one_col_list = one_col_list[skiprows:] if skiprows else one_col_list
        while '' in one_col_list:
            one_col_list.remove('')
    return one_col_list


def process_list_or_file(x):
    if isinstance(x, list) or isinstance(x, set):
        target_list = x
    else:
        if os.path.isfile(x):
            target_list = read_one_col_file(x)
        else:
            raise
    return target_list


def print_basename_in_dict(path_dict):
    for name, path in path_dict.items():
        print(f'{name}: {os.path.basename(path)}')


def check_path(path):
    print(f'{os.path.exists(path)} - {os.path.basename(path)}')


def check_path_in_dict(path_dict):
    print(f'Total {len(path_dict)} files')
    for name, path in path_dict.items():
        print(f'{os.path.exists(path)} - {name}: {os.path.basename(path)}')


def check_input_df(data, *args) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        if os.path.exists(data):
            df = pd.read_csv(data, *args)
        else:
            raise FileNotFoundError
    return df


def fill_path_dict(path_to_fill: str, fill_string: dict, exist_path_dict: dict = None):
    if exist_path_dict is None:
        path_dict = dict()
    else:
        path_dict = exist_path_dict.copy()

    for k, file_name in fill_string.items():
        file_name = [file_name] if isinstance(file_name, str) else file_name
        path_dict[k] = path_to_fill.format(*file_name)
    return path_dict


def join_path(path, *paths, create=False):
    pass


def write_inten_to_json(prec_inten: dict, file_path):
    total_prec = len(prec_inten)
    with open(file_path, 'w') as f:
        f.write('{\n')

        for prec_idx, (prec, inten_dict) in enumerate(prec_inten.items(), 1):
            f.write('    "%s": {\n' % prec)
            frag_num = len(inten_dict)
            for frag_idx, (frag, i) in enumerate(inten_dict.items(), 1):
                if frag_idx != frag_num:
                    f.write(f'        "{frag}": {i},\n')
                else:
                    f.write(f'        "{frag}": {i}\n')

            if prec_idx != total_prec:
                f.write('    },\n')
            else:
                f.write('    }\n')

        f.write('}')


def data_dump_load_skip(file_path, data=None, cover_data=False, update_file=False):
    if not os.path.exists(file_path):
        if data is not None:  # Here use 'is not None' because some thing will be wrong when the data is a pd.DataFrame. (Truth value is ambiguous error)
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise FileNotFoundError('No existing file and no input data')
    else:
        if data is not None:
            if update_file:
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
            elif cover_data:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                pass
        else:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
    return data
