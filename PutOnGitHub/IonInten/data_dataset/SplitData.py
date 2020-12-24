import pandas as pd
import os
import random
import copy
import ipdb
from collections import *
import numpy as np

DF_path = "/Users/admin/Documents/Project/IonIntensity/datasets/NewSpec"


def split_data(row_is_sample, data_path, train_ratio, dev_ratio, test_ratio=0, input_sep=None, to_csv=True, seed=666, ):
    """
    generate the training, developing and testing data using original data and random seed

    for ion intensity data stored in dict, it's suggested that set to_csv=False
    """

    random.seed(seed)
    output_dir = os.path.dirname(data_path)
    file_basename = os.path.splitext(os.path.basename(data_path))[0]
    file_ext = os.path.splitext(os.path.basename(data_path))[1]
    # ipdb.set_trace()
    file_suffix = ".csv" if to_csv else ".json"

    train_dst_path = os.path.join(output_dir, f'{file_basename}_train_no_holdout_{str(seed)}{file_suffix}')
    val_dst_path = os.path.join(output_dir, f'{file_basename}_val_no_holdout_{str(seed)}{file_suffix}')
    holdout_dst_path = os.path.join(output_dir, f'{file_basename}_holdout_{str(seed)}{file_suffix}')

    if file_ext == '.json':
        data = pd.read_json(data_path)
    else:
        if input_sep is not None:
            data = pd.read_csv(data_path, sep=input_sep)
        else:
            data = pd.read_csv(data_path)
    if row_is_sample:
        shuffled_data = data.sample(frac=1, random_state=seed)

        if test_ratio == 0:
            print(f"Will split {data_path} into {train_ratio}:{dev_ratio}")

            cut = int(len(data) * train_ratio / (train_ratio + dev_ratio))
            if to_csv:
                shuffled_data.iloc[:cut, :].to_csv(train_dst_path, index=True, index_label=False)
                shuffled_data.iloc[cut:, :].to_csv(val_dst_path, index=True, index_label=False)
            else:
                shuffled_data.iloc[:cut, :].to_json(train_dst_path)
                shuffled_data.iloc[cut:, :].to_json(val_dst_path)

        else:
            print(f"Will split {data_path} into {train_ratio}:{dev_ratio}:{test_ratio}")

            cut = int(len(shuffled_data) * train_ratio / (train_ratio + dev_ratio + test_ratio))
            cut2 = int(len(shuffled_data) * (train_ratio + dev_ratio) / (train_ratio + dev_ratio + test_ratio))
            if to_csv:
                shuffled_data.iloc[:cut, :].to_csv(train_dst_path, index=True, index_label=False)
                shuffled_data.iloc[cut:cut2, :].to_csv(val_dst_path, index=True, index_label=False)
                shuffled_data.iloc[cut2:, :].to_csv(holdout_dst_path, index=True, index_label=False)
            else:
                shuffled_data.iloc[:cut, :].to_json(train_dst_path)
                shuffled_data.iloc[cut:cut2, :].to_json(val_dst_path)
                shuffled_data.iloc[cut2:, :].to_json(holdout_dst_path)
    else:
        precursor = copy.deepcopy(data.columns.values)
        random.shuffle(precursor)
        if test_ratio == 0:
            # ipdb.set_trace()
            cut = int(len(precursor) * train_ratio / (train_ratio + dev_ratio))

            if to_csv:

                data.loc[:, precursor[:cut]].to_csv(train_dst_path, index=True, index_label=False)

                data.loc[:, precursor[cut:]].to_csv(val_dst_path, index=True, index_label=False)

            else:
                data.loc[:, precursor[:cut]].to_json(train_dst_path)

                data.loc[:, precursor[cut:]].to_json(val_dst_path)
        else:

            cut = int(len(precursor) * train_ratio / (train_ratio + dev_ratio + test_ratio))
            cut2 = int(len(precursor) * (train_ratio + dev_ratio) / (train_ratio + dev_ratio + test_ratio))
            if to_csv:
                data.loc[:, precursor[:cut]].to_csv(train_dst_path, index=True, index_label=False)

                data.loc[:, precursor[cut:cut2]].to_csv(val_dst_path, index=True, index_label=False)

                data.loc[:, precursor[cut2:]].to_csv(holdout_dst_path, index=True, index_label=False)
            else:

                data.loc[:, precursor[:cut]].to_json(train_dst_path)

                data.loc[:, precursor[cut:cut2]].to_json(val_dst_path)

                data.loc[:, precursor[cut2:]].to_json(holdout_dst_path)


def merge(phase_dir, data_name, data_type, out_dir=None):
    if data_type == "Ion":
        data_frames = [pd.read_json(path) for path in phase_dir]
        output_dir = os.path.dirname(data_frames[0])
        result = pd.concat(data_frames, axis=1, sort=False)
    else:
        result = None
        output_dir = None
        pass
    out_path = f'{data_name}_all.csv'
    if out_dir is None:

        result.to_csv(os.path.join(output_dir, out_path), index=True,
                      index_label=False)
    else:
        result.to_csv(os.path.join(out_dir, out_path), index=True,
                      index_label=False)


def split_jeff_raw_data(path, train_ratio, out, seed=666):
    random.seed(seed)
    Jeff = pd.read_csv(path, sep="\t")
    uniq = list(set(Jeff['IntPep']))
    random.shuffle(uniq)
    train = uniq[:int(train_ratio * len(uniq))]
    file_name = os.path.basename(path).split(".")[0]
    train_out_path = os.path.join(out, f'{file_name}_{str(train_ratio)}-train_{str(seed)}.csv')
    val_out_path = os.path.join(out, f'{file_name}_{str(1 - train_ratio)}-val_{str(seed)}.csv')
    Jeff.loc[Jeff['IntPep'].isin(train), :].to_csv(train_out_path, index=True, index_label=False)
    Jeff.loc[~Jeff['IntPep'].isin(train), :].to_csv(val_out_path, index=True, index_label=False)


def jeff_raw_clean(path):
    jeff = pd.read_csv(path)
    pep_stat = defaultdict(list)
    for pep, rt in zip(jeff['IntPep'], jeff["RT"]):
        pep_stat[pep].append(rt)

    seq = list(pep_stat.keys())
    rt_time = [np.median(rts) for rts in pep_stat.values()]
    data_frame = pd.DataFrame({"IntPep": seq, "RT": rt_time})
    ipdb.set_trace()
    out_path = path + "_remove_redundancy" + ".csv"
    data_frame.to_csv(out_path, index=True, index_label=False)


def merge_detect_data(pos_path, neg_path, data_name):
    pep_pos = pd.read_csv(pos_path, sep="\t", names=["sequence"])
    pep_neg = pd.read_csv(neg_path, sep="\t", names=["sequence"])
    pep_neg['detect'] = 0
    pep_pos['detect'] = 1
    output_dir = os.path.dirname(pos_path)
    result = pd.concat([pep_pos, pep_neg])
    out_path = f'{data_name}_all.csv'
    result.to_csv(os.path.join(output_dir, out_path), index=True,
                  index_label=False)


def main():
    # split_data(row_is_sample=True, data_path=DF_path, train_ratio=90, dev_ratio=10, test_ratio=0, input_sep="\t")
    # merge(["/Users/admin/Documents/Project/IonIntensity/datasets/acData/PhosDIA/20200820-Inten_Train-PhosDIA-Lib_DDA_Mod-seed0_811.json",
    #        "/Users/admin/Documents/Project/IonIntensity/datasets/acData/PhosDIA/20200820-Inten_Validation-PhosDIA-Lib_DDA_Mod-seed0_811.json",
    #        ],
    #       "DDAtrain_val", "/Users/admin/Documents/Project/IonIntensity/datasets/acData/PhosDIA/")
    # split_jeff_raw_data("/Users/admin/Documents/Project/RT/datasets/acData/20200724-Jeff-MQ_Author-Total_RT-1.txt", 0.95, "/Users/admin/Documents/Project/RT/datasets/acData/", seed=666)
    # jeff_raw_clean("/Users/admin/Documents/Project/RT/datasets/acData/20200724-Jeff-MQ_Author-Total_RT-1_0.05-val_666.csv")
    # jeff_raw_clean("/Users/admin/Documents/Project/RT/datasets/acData/20200724-Jeff-MQ_Author-Total_RT-1_0.95-train_666.csv")
    # for i in os.listdir(DF_path):
    #     if i.startswith("20201105"):
    #         split_data(row_is_sample=False, data_path=os.path.join(DF_path, i), train_ratio=90, dev_ratio=10,
    #                    test_ratio=0)
    split_data(row_is_sample=True, data_path="/Users/admin/Documents/Project/Detectability/Detect/Detectability_all_filtered_remove_unknown_token.csv",
               train_ratio=90, dev_ratio=10, test_ratio=0)
    # merge_detect_data("/Users/admin/Documents/Project/Detectability/Detect/20201110-Detect_Pos-iRTdb.txt",
    #                   "/Users/admin/Documents/Project/Detectability/Detect/20201110-Detect_Neg-PSP-mc2_7_30_max5_human.txt",
    #                   "Detectability")


if __name__ == '__main__':
    main()
