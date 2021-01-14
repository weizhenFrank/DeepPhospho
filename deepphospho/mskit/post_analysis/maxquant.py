import re
import pandas as pd


"""
检索结果的导入
    1. 每种文件单独
    2. 所有文件在一个文件夹中，读入指定文件关联 id

"""


def mq_modpep_to_intseq_1_6(x):
    x = x.replace('_', '')
    if '(Acetyl (Protein N-term))' in x:
        x = x.replace('(Acetyl (Protein N-term))', '')
        x = '*' + x
    else:
        x = '@' + x
    x = x.replace('C(Carbamidomethyl (C))', 'C')
    x = x.replace('M(Oxidation (M))', '1')
    x = x.replace('S(Phospho (STY))', '2')
    x = x.replace('T(Phospho (STY))', '3')
    x = x.replace('Y(Phospho (STY))', '4')
    return x


def mq_modpep_to_intseq_1_5(x):
    x = x.replace('_', '')
    if '(ac)' in x:
        x = x.replace('(ac)', '')
        x = '*' + x
    else:
        x = '@' + x
    x = x.replace('C(Carbamidomethyl (C))', 'C')
    x = x.replace('M(ox)', '1')
    x = x.replace('S(ph)', '2')
    x = x.replace('T(ph)', '3')
    x = x.replace('Y(ph)', '4')
    return x


def ion_info_from_mq(need_ion_type=('b', 'y'), verify_mass=True, verify_tolerance=20):
    pass


def inten_from_mq(x):
    ions = x['Matches']
    intens = x['Intensities']

    ion_intens_list = list(zip(ions.split(';'), intens.split(';')))

    inten_dict = dict()

    for ion_type in ['b', 'y']:
        ion_info = [_ for _ in ion_intens_list if _[0].startswith(ion_type)]

        current_num = 0
        ex_mod = 0
        for ion, inten in ion_info:

            ion_num = re.findall(f'{ion_type}(\d+)', ion)[0]
            ion_num = int(ion_num)

            re_charge = re.findall('\((\d)\+\)', ion)
            if re_charge:
                ion_charge = re_charge[0]
            else:
                ion_charge = '1'

            frag = f'{ion_type}{ion_num}+{ion_charge}'

            if '-' in ion:
                loss_type = re.findall('-(.+)$', ion)[0]
                frag += f'-1,{loss_type}'
                if ex_mod:
                    frag += f';{ex_mod},H3PO4'

            elif '*' in ion:
                if ex_mod == 0:
                    current_num = 0
                    ex_mod = 1
                else:
                    if ion_num <= current_num:
                        ex_mod += 1
                frag += f'-{ex_mod},H3PO4'

            else:
                frag += f'-Noloss'

            current_num = ion_num

            inten_dict[frag] = float(inten)

    return inten_dict


def show_ion_pair(result_df, iloc, return_df=True):
    ion_pairs = list(zip(result_df.iloc[iloc]['Masses'].split(';'),
                         result_df.iloc[iloc]['Matches'].split(';'),
                         result_df.iloc[iloc]['Intensities'].split(';')))
    if return_df:
        return pd.DataFrame(ion_pairs, columns=['FragMz', 'Frag', 'Inten'])
    else:
        for _ in ion_pairs:
            print(_)
    # pd.DataFrame(zip(jeff_mq_site_filter_df.loc[0]['Masses'].split(';'),
    # jeff_mq_site_filter_df.loc[0]['Matches'].split(';'),
    # jeff_mq_site_filter_df.loc[0]['Intensities'].split(';')))
