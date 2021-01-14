from .sn_constant import *
from .sn_utils import *

import re
import os
from collections import Counter
import prettytable
import pandas as pd

try:
    from sklearn.model_selection import train_test_split
except ModuleNotFoundError:
    pass

from mskit import rapid_kit
from mskit.calc import normalize_intensity


class SpectronautLibrarys(object):
    def __init__(self, lib_dict, init_oper=None):
        pass


class SpectronautLibrary(object):
    """
    self._library_storage is used for the original library dataframe storage so that self._lib_df can be restored when it is filtered.

    限定每列的 type 为 str，读取之后再转换指定列的格式
    """
    # TODO 提 fragment intensity 这种操作可以加并行，把提取步骤提出来，函数只负责做循环然后加一个装饰器？或是直接写两个方法

    LibBasicCols = [
        'PrecursorCharge',
        'ModifiedPeptide', 'StrippedPeptide',
        'iRT',
        'LabeledPeptide', 'PrecursorMz',
        'FragmentLossType', 'FragmentNumber', 'FragmentType', 'FragmentCharge', 'FragmentMz',
        'RelativeIntensity',
    ]
    FragInfoCols = [
        'FragmentLossType',
        'FragmentNumber',
        'FragmentType',
        'FragmentCharge',
        'FragmentMz',
        'RelativeIntensity',
    ]
    LibBasicCols_WithoutFrag = rapid_kit.subtract_list(LibBasicCols, FragInfoCols)

    def __init__(self, lib=None, spectronaut_version=14, pd_low_mem=False):
        self._spectronaut_version = spectronaut_version
        self._Mod = ModType(self._spectronaut_version)

        self._lib_path = None
        self._lib_df = None
        self._library_storage = None
        self._initial_lib_title = None

        self._lib_name = None

        self._prec = None
        self._modpep = None
        self._stripped_pep = None

        if lib is not None:
            self.set_library(lib, pd_low_mem=pd_low_mem)
            if isinstance(lib, str):
                self._lib_path = lib

    def init(self):
        pass

    def partial_init(self):
        pass

    def full_init(self):
        pass

    def require(self, required_list):
        require_dict = {
            'Prec': [self.add_prec()],
            'IntPep': [self.add_intpep()],
            'IntPrec': [self.add_prec(), self.add_intprec()]
        }
        return [require_dict[k] for k in required_list]

    def set_library(self, lib, pd_low_mem=False):
        if isinstance(lib, pd.DataFrame):
            self._lib_df = lib
        else:
            if os.path.isfile(lib):
                self._lib_path = lib
                self._lib_df = pd.read_csv(self._lib_path, sep='\t', low_memory=pd_low_mem)
            else:
                raise FileNotFoundError(f'The input library is not a dataframe and not a valid path.\n'
                                        f'Now is {type(lib)}: {lib}')

        self._initial_lib_title = self._lib_df.columns
        self._library_storage = self._lib_df.copy()

    def save(self, path, sep='\t', create_folder_if_notexist=False, save_func=None):
        """
        self.write_current_library
        """
        self._lib_df.to_csv(path, sep=sep, index=False)

    def __len__(self):
        return len(self._lib_df)

    def __add__(self, other):
        """
        Directly add the two libraries with no further operation
        """
        pass

    def __iadd__(self, other):
        pass

    def __mul__(self, other):
        """
        Let the fragments in two libraries combined together
        """
        pass

    def __getitem__(self, item):
        return self._lib_df[item]

    def __setitem__(self, key, value):
        self._lib_df[key] = value

    def columns(self):
        pass

    def fragment_loss_counts(self, col='FragmentLossType', readable_trans=True):
        loss_count = dict(Counter(self._lib_df[col]))
        if readable_trans:
            temp = dict()
            for loss_name, num in loss_count.items():
                if loss_name in LossType.SN_to_Readable:
                    temp[LossType.SN_to_Readable[loss_name]] = num
                else:
                    temp[loss_name] = num
            loss_count = temp.copy()
            del temp
        return loss_count

    def fragment_number_counts(self):
        pass

    def fragment_charge_counts(self):
        pass

    def summary(self, sum_list=None):
        summary_funcs = {
            'FragmentLossType': self.fragment_loss_counts,
        }
        summary_tabel_title = {
            'FragmentLossType': ['LossType', 'Number'],
        }
        long_sep = '-' * 20
        short_sep = '-' * 5
        if sum_list is None:
            sum_list = ['FragmentLossType', ]

        summary_contents = ''

        for each_feature in sum_list:
            sum_string = ''

            used_func = summary_funcs[each_feature]
            results = used_func()
            write_table = prettytable.PrettyTable(field_names=summary_tabel_title[each_feature])
            for each_result_name, result_num in results.items():
                write_table.add_row([each_result_name, result_num])

            sum_string += f'{long_sep}\n'
            sum_string += f'{short_sep} {each_feature} {short_sep}\n'
            sum_string += f'{write_table.get_string()}\n'
            sum_string += f'{long_sep}\n'
            print(sum_string)
            summary_contents += '\n'
            summary_contents += sum_string

    def backtrack_library(self):
        """
        replace lib_df with storage
        """
        self._lib_df = self._library_storage

    def store_current_library(self):
        self._library_storage = self._lib_df.copy()

    def switch_curr_store_lib(self):
        _ = self._lib_df.copy()
        self._lib_df = self._library_storage.copy()
        self._library_storage = _

    def add_prec(self):
        if 'Precursor' not in self._lib_df:
            self._lib_df['Precursor'] = self._lib_df['ModifiedPeptide'] + '.' + self._lib_df['PrecursorCharge'].astype(str)

    def add_intpep(self):
        # TODO This should be trans pep format (need a general function)
        self._lib_df['IntPep'] = self._lib_df['ModifiedPeptide'].apply(sn_modpep_to_intseq)

    def add_intprec(self):
        self._lib_df['IntPrec'] = self._lib_df['IntPep'] + '.' + self._lib_df['PrecursorCharge'].astype(str)

    def add_first_protein(self, target_col='ProteinGroups', add_col='FirstProtein'):
        def split_protein(x, col):
            acc = x[col]
            if pd.isna(acc):
                return np.nan
            acc = acc.split(';')[0].strip()
            return acc

        self._lib_df[add_col] = self._lib_df.apply(split_protein, col=target_col, axis=1)

    def add_protein_type(self, type_dict, target_col='ProteinGroups', add_col='ProteinType', add_method=lambda x: x.split(';')[0], split_sign=';'):
        """
        lambda x: x for first protein
        lambda x: x.split(';')[0] for ProteinGroups
        'All' to map type dict to all proteins in the cell. This needs the param split_sign to define the sign for splitting proteins.
        """
        pass

    def assign_protein(self, seq_dict, add_col='AssignedProteins'):
        pass

    def assign_protein_with_ref(self, ref_dict, seq_dict=None,
                                key_level='StrippedPeptide', add_col='AssignedProteins'):
        if seq_dict is not None:
            pass

    def trans_frag_losstype(self, trans_dict):
        self._lib_df['FragmentLossType_Raw'] = self._lib_df['FragmentLossType']
        self._lib_df['FragmentLossType'] = self._lib_df['FragmentLossType_Raw'].replace(trans_dict)

    def add_frag_name(self, trans_dict=None, used_losstype_col='FragmentLossType'):
        if trans_dict is not None:
            self.trans_frag_losstype(trans_dict)
        self._lib_df['FragName'] = self._lib_df["FragmentType"] + \
                                   self._lib_df["FragmentNumber"].astype(str) + '+' + \
                                   self._lib_df["FragmentCharge"].astype(str) + '-' + \
                                   self._lib_df[used_losstype_col]

    def add_inten_rank(self, base_col='Precursor', inten_col='RelativeIntensity', **rank_args):
        if 'method' not in rank_args:
            rank_args['method'] = 'min'
        self._lib_df['IntenRank'] = self._lib_df.groupby(base_col)[inten_col].rank(**rank_args).astype(int)

    def retain_basic_cols(self):
        self._lib_df = self._lib_df[self.LibBasicCols]

    def add_predefined_features(self, feature_list=None):
        """
        :param feature_list TODO 最后应该把所有feature单独添加，这里传一个list来选择添加哪些feature
        """
        self.add_prec()
        self.add_intpep()
        self.add_intprec()

    def remove_mod(self, mods, regex=False):
        mods = [mods] if isinstance(mods, str) else mods
        for mod in mods:
            self._lib_df = self._lib_df[~self._lib_df['ModifiedPeptide'].str.contains(mod, regex=regex)]

    def get_all_mods(self, sort=True, modpep_col='ModifiedPeptide'):
        mods = []
        for one_pep_mod in [re.findall(r'\[.+?\]', _) for _ in list(set(self._lib_df[modpep_col]))]:
            for mod in one_pep_mod:
                if mod not in mods:
                    mods.append(mod)
        if sort:
            mods = sorted(mods)
        return mods

    def get_all_aa(self, sort=True, used_col='StrippedPeptide'):
        aas = []
        for one_pep_aa in [list(_) for _ in list(set(self._lib_df[used_col]))]:
            for aa in one_pep_aa:
                if aa not in aas:
                    aas.append(aa)
        if sort:
            aas = sorted(aas)
        return aas

    def get_frag_inten(self, prec_col='Precursor') -> dict:
        lib_spec = dict()
        if 'FragName' not in self._lib_df.columns:
            self.add_frag_name()
        for prec, df in self._lib_df.groupby(prec_col):
            lib_spec[prec] = dict(df[['FragName', 'RelativeIntensity']].values)
        return lib_spec

    def get_rt_df(self, pep_col='ModifiedPeptide', rt_col='iRT') -> pd.DataFrame:
        rt_df = self._lib_df[[pep_col, rt_col]].drop_duplicates(pep_col)
        return rt_df

    def get_frag_inten_phos(self, prec_col='Precursor', output_key_col=None,
                            limit_charge=(1, 2),
                            limit_loss=('noloss', 'H3PO4'),
                            index_loss='H3PO4', min_frag_num=None) -> dict:
        """
        这个现在只用来作为 pDeep2 的 test 时提取 library 中的 Intensity
        如果是用于 training 提取 library 中的 fragment Intensity 使用 get_frag_inten
        之后应该把这个里面对 Phos loss 最小 by 离子位置的功能放在外面
        """
        lib_spec = dict()
        if 'FragName' not in self._lib_df.columns:
            self.add_frag_name()
        if output_key_col is None:
            output_key_col = prec_col
        if min_frag_num is None:
            min_frag_num = 0
        for prec, df in self._lib_df.groupby(prec_col):
            modpep, charge = prec.split('.')
            output_key = df.iloc[0][output_key_col]
            strip_pep, mod_sites, mods = rapid_kit.substring_finder(modpep.replace('_', ''))
            phos_mod_site = [mod_sites[i] for i, mod in enumerate(mods) if mod == '[Phospho (STY)]']
            st_phosmod_site = [s for s in phos_mod_site if strip_pep[s - 1] in ('S', 'T')]
            if st_phosmod_site:
                positive_min_phos_site = min(st_phosmod_site)
                negetive_min_phos_site = len(strip_pep) - max(st_phosmod_site) + 1
            else:
                positive_min_phos_site = len(strip_pep) + 1
                negetive_min_phos_site = len(strip_pep) + 1
            frag_dict = dict()
            for row_index, row in df.iterrows():
                frag_type = row["FragmentType"]
                frag_num = row["FragmentNumber"]
                frag_charge = row["FragmentCharge"]
                frag_losstype = row["FragmentLossType"]
                if frag_charge not in limit_charge:
                    continue
                if frag_losstype not in limit_loss:
                    continue
                if frag_losstype == index_loss:
                    if frag_type == 'b' and frag_num < positive_min_phos_site:
                        continue
                    elif frag_type == 'y' and frag_num < negetive_min_phos_site:
                        continue
                    else:
                        pass
                frag_dict[row['FragName']] = row['RelativeIntensity']
            if len(frag_dict) >= min_frag_num:
                lib_spec[output_key] = normalize_intensity(frag_dict, max_num=100)
        return lib_spec

    def remove_library_add_col(self):
        new_cols = [_ for _ in self._lib_df.columns if _ in self._initial_lib_title]
        self._lib_df = self._lib_df[new_cols]

    def return_library(self, remove_add_col=False):
        if remove_add_col:
            self.remove_library_add_col()
        return self._lib_df.copy()

    to_df = return_library

    def write_current_library(self, output_path, remove_add_col=True):
        """
        write current lib to file
        """
        if remove_add_col:
            self.remove_library_add_col()
        self._lib_df.to_csv(output_path, index=False, sep='\t')

    def split_library(self, split_size=(0.9, 0.1), split_by='ModifiedPeptide', save=None, seed=0):
        by_list = self._lib_df[split_by].drop_duplicates().tolist()
        if len(split_size) == 2:
            train_list, test_list = train_test_split(by_list, train_size=split_size[0], test_size=split_size[1], random_state=seed)
            if save is None:
                return (self._lib_df[self._lib_df[split_by].isin(train_list)],
                        self._lib_df[self._lib_df[split_by].isin(test_list)])
            else:
                pass
        elif len(split_size) == 3:
            size_outof_train = split_size[1] + split_size[2]
            train_list, val_test_list = train_test_split(by_list, train_size=split_size[0], test_size=size_outof_train, random_state=seed)
            val_size, test_size = split_size[1] / size_outof_train, split_size[2] / size_outof_train
            val_list, test_list = train_test_split(val_test_list, train_size=val_size, test_size=test_size, random_state=seed)
            return (self._lib_df[self._lib_df[split_by].isin(train_list)],
                    self._lib_df[self._lib_df[split_by].isin(val_list)],
                    self._lib_df[self._lib_df[split_by].isin(test_list)])
        else:
            pass

    def library_basic_info(self):
        """
        Count basic library infomation includes
            'protein_group', 'protein', 'modpep', 'prec', 'stripped_pep'
        :return:
        """
        protein_group = set(self._lib_df['ProteinGroups'])
        protein = set(sum([_.split(';') for _ in protein_group], []))
        self._modpep = set(self._lib_df['ModifiedPeptide'])
        self.add_prec()
        self._prec = set(self._lib_df['Precursor'])
        self._stripped_pep = set(self._lib_df['StrippedPeptide'])
        library_info_dict = dict((('ProteinGroup', len(protein_group)),
                                  ('SingleProtein', len(protein)),
                                  ('ModifiedPeptide', len(self._modpep)),
                                  ('Precursor', len(self._prec)),
                                  ('StrippedPeptide', len(self._stripped_pep)),
                                  ))
        charge = self._lib_df['PrecursorCharge'].tolist()
        return library_info_dict

    def strippep_length_distrib(self) -> pd.Series:
        """
        Count peptides length in the library
        :return: pandas.Series, which has all peptide length as index and corresponding peptide number as value. The index is sorted increasely.
        """
        stripped_pep = self._lib_df['StrippedPeptide'].drop_duplicates()
        stripped_pep_len = stripped_pep.str.len()
        len_ser = stripped_pep_len.value_counts()
        sorted_len_ser = len_ser.sort_index()
        return sorted_len_ser

    def strippep_miss_cleavage_distrib(self) -> pd.Series:
        """
        Count miss cleavage in the library
        :return: pandas.Series, which has miss cleavage number as index and corresponding peptide number as value. The index is sorted increasely.
        """
        stripped_pep = self._lib_df['StrippedPeptide'].drop_duplicates()
        mc_ser = stripped_pep.str.count('[KR].*?') - stripped_pep.str.count('.+[KR]$')
        mc_count = mc_ser.value_counts()
        sorted_mc_count = mc_count.sort_index()
        return sorted_mc_count

    def prec_charge_distrib(self) -> pd.Series:
        """
        Count charge in the library
        :return: pandas.Series, which has charge state as index and corresponding peptide number as value. The index is sorted increasely.
        """
        prec_nonredundant_df = self.get_nonredundant_prec_df(self._lib_df)
        prec_charge = prec_nonredundant_df['PrecursorCharge']
        charge_count = prec_charge.value_counts()
        sorted_charge_count = charge_count.sort_index()
        return sorted_charge_count

    def library_ms1_intensity_distrib(self) -> pd.Series:
        """
        MS1 response of each precursor
        :return:
        """
        prec_nonredundant_df = self.get_nonredundant_prec_df(self._lib_df)
        prec_ms1_response = prec_nonredundant_df['ReferenceRunMS1Response']
        return prec_ms1_response

    def library_oxidation_distrib(self):
        """
        :return: Two pd.Series, which are oxidation count and methionine count, respectively
        """
        stripped_pep = self._lib_df['ModifiedPeptide'].drop_duplicates()
        oxi_ser = stripped_pep.str.count(r'M\[Oxi.*?')
        oxi_count = oxi_ser.value_counts()
        sorted_oxi_count = oxi_count.sort_index()
        return sorted_oxi_count

    def library_length_charge_distrib(self) -> pd.Series:
        pass

    def filter_length(self, min_len=None, max_len=None, col='StrippedPeptide'):
        self._lib_df = self._lib_df[self._lib_df[col].str.len().apply(lambda x: min_len <= x <= max_len)]

    def filter_prec_charge(self, charge=(1, 2, 3, 4, 5), prec_charge_col='PrecursorCharge'):
        if isinstance(charge, int):
            charge = (charge,)
        self._lib_df = self._lib_df[self._lib_df[prec_charge_col].astype(int).isin(charge)]

    def filter_peak_losstype(self, min_noloss_peak: int = 6, loss_type='Noloss',
                             prec_col='Precursor', loss_col='FragmentLossType'):
        self._lib_df = self._lib_df.loc[self._lib_df.groupby(prec_col)[loss_col].filter(
            lambda x: x.value_counts().get(loss_type, 0) >= min_noloss_peak).index]

    def filter_library(self, min_len=None, max_len=None, min_noloss_peaks: int = 6, charge=(1, 2, 3, 4, 5)):
        """
        For psm filter, to get get_one_prefix_result library with better quantity for training.
        This will filter the library with three conditions:
            1. stripped peptide length: from min length to 95% max length of whole library length distribution,
            which means the max length range is set as the length when cumulation of sorted pep length num achieves 95% of total pep num.
            The min length is usually set to 7 since the default set in spectronaut is 7.
            The max length can be calculated automatically with ##########################
            2. fragment loss type: 'noloss' type should equals or more than 6 as default
            3. precursor charge: this will be 1-5 in DIA
            Additional condition: 4. miss cleavage: 0-1, this is not so useful

        Either min_len or max_len is not None, the length filter will be done.
        :param min_len: The min length of stripped peptide
        :param max_len: The max length of stripped peptide. An int number will lock the max length to it,
        whill a float number will make the max length an expected number to cover certein percentage of total peptides.
        Example: max_len=26 means the max length is 26, and max_len=0.95 means the length covers 95% of the total peptides will be max length
        :param min_noloss_peaks:
        :param charge:
        """
        if isinstance(charge, int):
            charge = (charge,)

        if isinstance(max_len, float):
            len_distrib = self.strippep_length_distrib()
            if min_len:
                len_distrib = len_distrib[len_distrib.index >= min_len]
            accu_num = 0
            targeted_pep_num = sum(len_distrib) * max_len
            for _l, _n in len_distrib.items():
                accu_num += _n
                if accu_num >= targeted_pep_num:
                    max_len = _l
                    break

        def length_filter(x):
            if min_len:
                if len(x) < min_len:
                    return False
            if max_len:
                if len(x) > max_len:
                    return False
            return True

        length_filtered_df = self._lib_df[self._lib_df['StrippedPeptide'].apply(
            length_filter)] if any((min_len, max_len)) else self._lib_df.copy()  # filter_condition 1

        charge_filtered_df = length_filtered_df[
            length_filtered_df['PrecursorCharge'].astype(int).isin(charge)] if charge else length_filtered_df  # filter_condition 3

        if min_noloss_peaks:
            peak_count = charge_filtered_df.groupby('Precursor')['FragmentLossType'].transform(lambda x: len(x[x == 'noloss']))
            noloss_only_df = charge_filtered_df.loc[peak_count[peak_count >= min_noloss_peaks].index]
            self._lib_df = noloss_only_df
        else:
            self._lib_df = charge_filtered_df

    def library_keep_main_cols(self):
        """
        Remove some columns that are not necessary in the library
        """
        self._lib_df = self._lib_df[SNLibraryTitle.LibraryMainCol]

    def library_keep_main_cols_pgout(self):
        """
        Keep the same columns as library_keep_main_cols except 'ProteinGroups'
        """
        self._lib_df = self._lib_df[SNLibraryTitle.LibraryMainColPGOut]

    def get_prec_list(self):
        if self._prec:
            return list(self._prec)
        else:
            if 'Precursor' not in self._lib_df:
                self.add_prec()
            prec_list = self._lib_df['Precursor'].drop_duplicates().tolist()
            self._prec = prec_list
            return prec_list

    def get_modpep_list(self):
        if self._modpep:
            return list(self._modpep)
        else:
            modpep_list = self._lib_df['ModifiedPeptide'].drop_duplicates().tolist()
            self._modpep = modpep_list
            return modpep_list

    def extract_rt_data(self, pep_col='ModifiedPeptide', rt_col='iRT', return_type='dict'):
        """
        This will extract nonredundancy modpep and corresponding rt
        :return: dict like {'modpep': rt, ...}, or list like [(modpep, rt), ...]
        """
        return get_lib_rt_info(self._lib_df, pep_col, rt_col, return_type)

    def extract_fragment_data(self, norm=False):
        """
        :param norm False means no normalization and a number means the max number of intensity
        """
        frag_dict = get_lib_fragment_info(self._lib_df, norm=norm)
        return frag_dict

    def library_keep_target(self, target_protein_list):
        """
        Keep psms that contain target protein
        target protein may be consist of protein groups, which are separated by semicolons
        :return:
        """
        target_list = rapid_kit.process_list_or_file(target_protein_list)
        target_set = set(target_list)
        self._lib_df = self._lib_df[self._lib_df.apply(rapid_kit.protein_groups_match, col='ProteinGroups', args=(target_set,), axis=1)]

    def library_remove_target(self, target_protein_list):
        """
        Remove psms that contain target protein
        target protein may be consist of protein groups, which are separated by semicolons
        196 ms ± 97.3 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
        :return:
        """
        target_list = rapid_kit.process_list_or_file(target_protein_list)
        target_set = set(target_list)
        self._lib_df = self._lib_df[~self._lib_df.apply(rapid_kit.protein_groups_match, col='ProteinGroups', args=(target_set,), axis=1)]

    def library_remove_target_re(self, target_protein_list):
        """
        Remove psms that contain target protein
        target protein may be consist of protein groups, which are separated by semicolons
        1min 56s ± 199 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        Use library_remove_target instead
        """
        if os.path.isfile(target_protein_list):
            with open(target_protein_list, 'r') as target_handle:
                target_list = [_.strip('\n') for _ in target_handle.readlines()]
                while '' in target_list:
                    target_list.remove('')
        else:
            target_list = target_protein_list
        match_target = re.compile('|'.join(['(' + _ + ')' for _ in target_list]))
        self._lib_df = self._lib_df[~(self._lib_df['ProteinGroups'].str.contains(match_target))]

    def check_col_format(self, start_with='_', start_with_num=1, end_with='_', end_with_num=1, normal_char='[A-Z]', special_char=('[', ']'), exclude_char=''):
        pass

    def find_col_format(self):
        pass

    @staticmethod
    def get_nonredundant_prec_df(lib_df: pd.DataFrame) -> pd.DataFrame:
        return lib_df.drop_duplicates(('PrecursorCharge', 'ModifiedPeptide'))

    @staticmethod
    def get_nonredundant_precursor_index(lib_df: pd.DataFrame, position=1) -> pd.Index:
        """
        Consider get_one_prefix_result psm as get_one_prefix_result block.
        :param lib_df:
        :param position: 1 means block start index, -1 means block end index
        :return: pd.Index, which is the first line index or the end line index of each psm block
        """
        precursor_block_index = lib_df.loc[(lib_df['PrecursorCharge'].astype(str) +
                                            lib_df['ModifiedPeptide']) !=
                                           (lib_df['PrecursorCharge'].astype(str).shift(position) +
                                            lib_df['ModifiedPeptide'].shift(position))].index
        return precursor_block_index

    @staticmethod
    def get_psm_block_index(lib_df: pd.DataFrame) -> list:
        """
        Consider get_one_prefix_result psm as get_one_prefix_result block. This can return get_one_prefix_result list contains index tuples for library dataframe slice
        If the first precursor occupy the first 9 lines (which means line 0-8), the tuple will be (0, 9), then library dataframe can be sliced directly.
        :return: list of tuples, like [(0, 9), (9, 13), (13, 19), ...]
        """
        precursor_block_start_index = SpectronautLibrary.get_nonredundant_precursor_index(lib_df, position=1)
        precursor_block_end_index = SpectronautLibrary.get_nonredundant_precursor_index(lib_df, position=-1)
        block_index_list = list(zip(precursor_block_start_index, precursor_block_end_index))
        return block_index_list
