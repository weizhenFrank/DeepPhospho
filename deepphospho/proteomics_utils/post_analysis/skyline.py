import pandas as pd
import numpy as np
import os
import scipy.stats
import time
import itertools
import argparse
import matplotlib.pyplot as plt
from sys import argv


def read_need_list(list_content):
    if os.path.isfile(list_content):
        with open(list_content, 'r') as f:
            need_list = f.read().split('\n')
    elif isinstance(list_content, str):
        need_list = list_content.split('\n')
    else:
        need_list = list_content
    need_list = [_.replace('\n', '') for _ in need_list]
    while '' in need_list:
        need_list.remove('')
    need_list = [_.replace('C', 'C[+57.021464]') for _ in need_list]
    return need_list


def process_transition_file(transition_file, need_list=None):
    transition_df = pd.read_csv(transition_file, header=-1)
    if need_list:
        transition_df = transition_df[transition_df[3].isin(need_list)]

    transition_df = transition_df.drop(transition_df.loc[transition_df[5] == transition_df[5].shift(1)].index)

    fragment_order_12_list = ['b1', 'b2', 'y1', 'y2']
    transition_df = transition_df[~(transition_df[5].isin(fragment_order_12_list))]

    return transition_df


if __name__ == '__main__':
    trans_file = argv[1]
    nl = read_need_list(argv[2])
    trans_df = process_transition_file(trans_file, nl)
    out_path = os.path.splitext(trans_file)[0] + '-Processed.csv'
    trans_df.to_csv(out_path, index=False, header=False)


def transition_list_filter(raw_file, pep_list):
    pass


def read_skyline_result(sky_file, sample_identi=None):
    result_df = pd.read_csv(sky_file)
    stripped_pep_list = result_df['Peptide Sequence'].drop_duplicates().dropna().tolist()
    samp_list = result_df['Replicate Name'].drop_duplicates().dropna().tolist()
    df_columns = result_df.columns

    if 'Area' not in df_columns:
        use_total_fragment_area = True
    else:
        use_total_fragment_area = False

    if sample_identi:
        samp_list = [_ for _ in samp_list if sample_identi[0].lower() in _.lower()] + [_ for _ in samp_list if sample_identi[1].lower() in _.lower()]
    sample_num = len(samp_list)
    pep_num = len(stripped_pep_list)
    quant_df = pd.DataFrame(np.zeros((pep_num, sample_num)), index=stripped_pep_list, columns=samp_list)

    for each_stripep in stripped_pep_list:
        each_pep_df = result_df[result_df['Peptide Sequence'] == each_stripep]
        each_prec_df = each_pep_df[each_pep_df['Fragment Ion'] == 'precursor']
        each_fragment_df = each_pep_df[~(each_pep_df['Fragment Ion'] == 'precursor')]

        for each_rep in samp_list:
            if use_total_fragment_area:
                each_fragment_area_sum = each_prec_df[each_prec_df['Replicate Name'] == each_rep]['Total Area Fragment'].iloc[0]
            else:
                try:
                    coeluting_true_df = each_fragment_df[each_fragment_df['Coeluting'] == True]
                except KeyError:
                    coeluting_true_df = each_fragment_df
                each_rep_df = coeluting_true_df[coeluting_true_df['Replicate Name'] == each_rep]
                try:
                    each_fragment_area_sum = each_rep_df['Area'].sum()
                except:
                    raise
            quant_df.loc[each_stripep, each_rep] = each_fragment_area_sum

    for _ in range(sample_num):
        quant_df.iloc[:, _] /= align_list[_]
    return quant_df


def drop_rep(df, method='StandardScore'):
    samp_list = df.columns
    sample_num = len(samp_list)
    rep_num = int(sample_num / 2)

    df1 = df.loc[:, samp_list[0: rep_num]]
    df2 = df.loc[:, samp_list[rep_num: rep_num * 2]]

    df1 = df1.drop(calc_df_ss_sum(df1).idxmax(), axis=1)
    df2 = df2.drop(calc_df_ss_sum(df2).idxmax(), axis=1)
    return pd.merge(df1, df2, left_index=True, right_index=True)


def calc_df_ss_sum(df):
    """
    增加参数：
        剩余的rep数量
        每个肽段的权重
    """

    def calc_standard_score(x):
        m = np.mean(x)
        s = np.std(x, ddof=1)
        _ = x - m
        _ = _ / s
        return _

    standard_score_df = df.apply(calc_standard_score, axis=1).applymap(abs)
    ss_sum = standard_score_df.apply(lambda x: sum([_ for _ in x if _ != x.max()]), axis=0)
    return ss_sum


def calc_all_ratio(df1, df2, method='mean'):
    if df1.shape != df2.shape:
        raise
    nrow, ncol = df1.shape
    ratio_matrix = np.zeros((nrow, ncol ** 2))
    for row_num in range(nrow):
        col_num = 0
        for col_num1 in range(ncol):
            for col_num2 in range(ncol):
                ratio_matrix[row_num, col_num] = df2.iloc[row_num, col_num2] / df1.iloc[row_num, col_num1]
                col_num += 1
    return ratio_matrix.mean(axis=1), ratio_matrix.std(axis=1, ddof=1)


def stats_within_group(quant_df, sample_identi):
    samp_list = quant_df.columns
    sample_num = len(samp_list)
    rep_num = int(sample_num / 2)

    # MeanArea
    quant_df[f'MeanArea-{sample_identi[0]}'] = quant_df.loc[:, samp_list[0: rep_num]].apply(np.mean, axis=1)
    quant_df[f'MeanArea-{sample_identi[1]}'] = quant_df.loc[:, samp_list[rep_num: rep_num * 2]].apply(np.mean, axis=1)
    # Ratio-respective
    for rep_series in range(rep_num):
        each_bi = quant_df.iloc[:, rep_series + rep_num] / quant_df.iloc[:, rep_series]
        quant_df[f'Ratio-{rep_series + 1}'] = each_bi
    # Ratio-Mean
    quant_df[f'Ratio-Mean'] = quant_df.loc[:, [_ for _ in quant_df.columns if 'Ratio' in _]].apply(np.mean, axis=1)
    # STD-Ratio
    quant_df[f'STD-Ratio'] = quant_df.loc[:, [_ for _ in quant_df.columns if 'Ratio' in _]].apply(np.std, ddof=1, axis=1)
    # CV-Ratio
    quant_df[f'CV-Ratio'] = quant_df[f'STD-Ratio'] / quant_df[f'Ratio-Mean']
    # p-value
    pvalue_list = []
    for each_pep in quant_df.index:
        array_t_test = scipy.stats.ttest_ind(quant_df.loc[each_pep, samp_list[0: rep_num]], quant_df.loc[each_pep, samp_list[rep_num: rep_num * 2]], equal_var=True)
        float_p_value = array_t_test[1]
        if scipy.stats.levene(quant_df.loc[each_pep, samp_list[0: rep_num]], quant_df.loc[each_pep, samp_list[rep_num: rep_num * 2]])[1] < 0.05:
            array_t_test = scipy.stats.ttest_ind(quant_df.loc[each_pep, samp_list[0: rep_num]], quant_df.loc[each_pep, samp_list[rep_num: rep_num * 2]], equal_var=False)
            float_p_value = array_t_test[1]
        pvalue_list.append(float_p_value)
    quant_df[f'P-value'] = pvalue_list
    # MeanRatio
    mean_multi_ratio, std_multi_ratio = calc_all_ratio(quant_df.loc[:, samp_list[0: rep_num]], quant_df.loc[:, samp_list[rep_num: rep_num * 2]])
    quant_df['MeanRatio'] = mean_multi_ratio
    quant_df['STD-MeanRatio'] = std_multi_ratio
    # STD
    quant_df[f'STD-{sample_identi[0]}'] = quant_df.loc[:, samp_list[0: rep_num]].apply(np.std, ddof=1, axis=1)
    quant_df[f'STD-{sample_identi[1]}'] = quant_df.loc[:, samp_list[rep_num: rep_num * 2]].apply(np.std, ddof=1, axis=1)
    # CV
    quant_df[f'CV-{sample_identi[0]}'] = quant_df[f'STD-{sample_identi[0]}'] / quant_df[f'MeanArea-{sample_identi[0]}'] * 100
    quant_df[f'CV-{sample_identi[1]}'] = quant_df[f'STD-{sample_identi[1]}'] / quant_df[f'MeanArea-{sample_identi[1]}'] * 100

    return quant_df


def write_result(input_file, quant_df, quant_df2=None):
    out_path = os.path.splitext(input_file)[0] + '-ProcessedAligned.xlsx'

    with pd.ExcelWriter(out_path) as f:
        quant_df.style.applymap(lambda _: 'font-family: Aria').to_excel(f, sheet_name='RawData')
        quant_df2.style.applymap(lambda _: 'font-family: Aria').to_excel(f, sheet_name='RepDrop')


if __name__ == '__main__':
    result_file = r'.\Result.csv'
    sample_identifier = 'Control,Experiment'
    align_list = [1.1, 1.1, 1.1, 1.1] + [1.1, 1.1, 1.1, 1.1]
    sample_identifier = sample_identifier.split(',')
    raw_result_df = read_skyline_result(result_file, sample_identifier)
    rep_drop_df = drop_rep(raw_result_df)
    processed_result_df = stats_within_group(raw_result_df, sample_identifier)
    processed_result_drop_df = stats_within_group(rep_drop_df, sample_identifier)
    write_result(result_file, processed_result_df, processed_result_drop_df)
