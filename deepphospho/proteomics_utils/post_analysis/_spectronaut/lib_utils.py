import re
import numpy as np
import pandas as pd

from deepphospho.proteomics_utils import rapid_kit as rk


def substract_snlib(snlib_1, snlib_2, col='Prec'):
    """
    将第一个 lib 按给定的 col 删除第二个 lib 中的内容
    这里两个 lib 的大小不一定哪边一定大，只是删除 overlap 的部分
    """
    pass


def add_pep_pos_in_prot(x, seq_dict: dict):
    """
    如果在一个 protein 匹配到了多个位置，用 / 分隔 （result 中的 PEP.PeptidePosition 是用 , 分隔在同一蛋白多次匹配的情况）
    如果一个 protein group 里有多个蛋白，用 ; 分隔
    如果 protein group 中所有 protein 都没有匹配到，则返回 na
    如果 protein group 中有部分匹配到，则没有匹配到的部分为空 ''，但不影响其他匹配到的位置，如 '54;;' 中的后两个蛋白均没有匹配到
    """
    pep_pos = []
    pep = x['StrippedPeptide']
    if pd.isna(x['ProteinGroups']):  # 如果 result 中找到的肽段没有 assign 蛋白
        return np.nan
    for protein in x['ProteinGroups'].split(';'):
        if protein not in seq_dict:
            pep_pos.append('')
        else:
            protein_seq = seq_dict[protein]
            sites = ''
            for site_index, re_result in enumerate(re.finditer(pep, protein_seq)):
                sites += f'{str(re_result.start() + 1)}/'
            pep_pos.append(sites.strip('/'))
    str_pep_pos = ';'.join(pep_pos)
    if str_pep_pos.strip(';') == '':
        str_pep_pos = np.nan
    return str_pep_pos


def add_phossite_in_pep(x):
    """
    如果一条肽段有多个修饰位点，用 ; 分隔
    """
    if x['IsPhosPep']:
        pep = x['ModifiedPeptide'].replace('_', '')
        sites = []
        for mod_pos, mod in zip(*rk.substring_finder(pep)[1:]):
            if mod == '[Phospho (STY)]':
                sites.append(mod_pos)
        if sites:
            return ';'.join(map(str, sites))
        else:
            return np.nan
    else:
        return np.nan


def add_phossite_in_prot(x):
    """
    如果一条肽段分配给的 protein group 有多个蛋白，用 ; 分隔
    如果一条肽段有多个位点，用 , 分隔
    蛋白 id 和位点间用 - 连接
    如果一条肽段可以匹配到一个蛋白的多个位置，用 / 分隔
    如：P1-142,P1-145/P1-275,P1-278;P2-51,P2-54
        这条肽段的 protein groups 为两个蛋白 P1 和 P2
        这条肽段在 P1 上可以同时作为第 140 位氨基酸开始或 273 位氨基酸开始，其中磷酸化位点在肽段上为 3 位和 6 位，则蛋白序列上的磷酸化位点可以为 142+145 或 275+278 两种情况
        而在 P2 这个蛋白的序列中，这条肽段只出现了一次，肽段在蛋白序列的位置是从 49 开始，则两个位点在蛋白的位置为 51 和 54
    """
    if pd.isna(x['PepPhosSite']) or pd.isna(x['PepPos']) or pd.isna(x['ProteinGroups']):
        return np.nan
    pep_phos_site = x['PepPhosSite'].split(';')
    prot_site = ''
    try:
        for protein, pep_pos in zip(x['ProteinGroups'].split(';'), x['PepPos'].split(';')):  # 如果一个 protein group 有多个 protein
            if pep_pos == '':
                prot_site += ';'
                continue
            for one_pep_pos in pep_pos.split('/'):  # 如果一条肽段匹配到了同个 protein sequence 上的不同位置
                for phos_site in pep_phos_site:  # 如果一条肽段有多个 phosphosite
                    prot_site += f'{protein}-{int(one_pep_pos) + int(phos_site) - 1},'
                prot_site = prot_site.strip(',')
                prot_site += '/'
            prot_site = prot_site.strip('/')
            prot_site += ';'
        prot_site = prot_site[:-1]
    except:
        print(pep_pos, phos_site)
        print(x['Precursor'])
        raise
    return prot_site
