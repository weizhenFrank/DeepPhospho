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
        one_col_list = [_.strip('\n') for _ in f.readlines()]
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


def data_dump_load_skip(file_path, data=None, update=False):
    if not os.path.exists(file_path):
        if data is not None:  # Here use 'is not None' because some thing will be wrong when the data is a pd.DataFrame. (Truth value is ambiguous error)
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise FileNotFoundError('No existing file and no input data')
    else:
        if data is not None:
            if update:
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
            else:
                pass
        else:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
    return data
