from mskit import rapid_kit

import os
from tqdm import tqdm

import pandas as pd
import numpy as np


def processing_progress_logger(progress_type=None, print_progress=True, save_log=False):
    pass


class SpectronautResults(object):
    """
    This will store multi results as a dict
    """
    def __init__(self, result_path_dict=None, read_within_init=False, ident_func=lambda x: os.path.basename(x)):  # 这里可以是list然后传一个ident func来拿到result name或dict直接给定result name
        self.result_path_dict = result_path_dict
        self.filename2key_func = ident_func
        self.results = dict()

        self.total_result_file_num = 0

        if read_within_init:
            self.read_results()

        self.total_result_data = None
        self.phos_result_data = None

        self.stats_number_matrix = None
        self.stats_number_df = None

    def __len__(self):
        return len(self.results)

    def read_results(self, result_path_dict=None, print_progress=False):
        if result_path_dict:
            self.result_path_dict = result_path_dict
        self.total_result_file_num = len(self.result_path_dict)
        for i, (name, f) in enumerate(self.result_path_dict.items()):
            if print_progress:
                print(f'Loading result file {i+1}/{self.total_result_file_num}')
            self.results[name] = SpectronautResult(f)

    def remove_all_results(self):
        self.results = dict()

    def add_predefined_features_to_all(self, ptm_thres=0.99, print_progress=False):
        for name, result_df in self.results.items():
            result_df.add_predefined_features(ptm_thres=ptm_thres)

    def get_results_stats(self):
        self.total_result_data = dict()
        self.phos_result_data = dict()
        for col in ['FirstProtein', 'Prec', 'EG.ModifiedPeptide', 'ProtPhosPosComb']:
            self.total_result_data[col] = dict()
            self.phos_result_data[col] = dict()
            for lib_type, sn_result in self.results.items():
                result_df = sn_result.return_result_df()
                if col == 'ProtPhosPosComb':
                    self.total_result_data[col][lib_type] = set(';'.join(result_df[col].dropna().tolist()).split(';'))
                    self.phos_result_data[col][lib_type] = set(';'.join(result_df[result_df['IsPhospep']][col].dropna().tolist()).split(';'))
                else:
                    self.total_result_data[col][lib_type] = set(result_df[col])
                    self.phos_result_data[col][lib_type] = set(result_df[result_df['IsPhospep']][col])

    def construct_result_stats_matrix(self):
        self.stats_number_matrix = np.array([
            sum([
                [
                    len(self.total_result_data[col][lib_type]),
                    len(self.phos_result_data[col][lib_type])
                ] for col in ['FirstProtein', 'Prec', 'EG.ModifiedPeptide', 'ProtPhosPosComb']
            ], []) for lib_type in self.results.keys()
        ])

    def construct_result_stats_df(self):
        self.stats_number_df = pd.DataFrame(
            self.stats_number_matrix[:, :-1],
            index=list(self.results.keys()),
            columns=sum([[_, _ + '-PhosOnly'] for _ in ['Protein', 'Prec', 'Modpep', 'Phossite']], [])[:-1],
        )

    def return_result_dfs(self):
        return self.results


class SpectronautResult(object):
    def __init__(self, result_file=None):

        # The path of result file and read the result file if the file path is passed when news a object of this class
        self.result_file = result_file
        if self.result_file:
            self.read_result(self.result_file)
        else:
            self.result_df = None

        self.initial_cols = None

        self.decoy_on = False  # if IsDecoy in the columns and True in the values -> decoy_on = True

    def read_result(self, result_file):
        """
        :param result_file The path of result file from SN (Normal scheme)
        This func can be called any time to set a result file.
        And the func init_results will also be called at first, then the previously stored varibles will be removed.
        """
        self.init_results()
        if isinstance(result_file, pd.DataFrame):
            self.result_df = result_file
        else:
            self.result_file = result_file
            self.result_df = pd.read_csv(result_file, sep='\t', low_memory=False)
        self.initial_cols = self.result_df.columns

    def init_results(self):
        """
        This will set all stored contants and results to None
        """
        self.result_file = None
        self.result_df = None
        self.initial_cols = None

    def add_first_protein(self):
        self.result_df['FirstProtein'] = self.result_df['PG.ProteinGroups'].apply(lambda x: x.split(';')[0])

    def add_prec(self):
        self.result_df['Prec'] = self.result_df['EG.ModifiedPeptide'] + '.' + self.result_df['FG.Charge'].astype(int).astype(str)

    def add_intseq(self):
        str_mod_to_int_dict = {
            'C[Carbamidomethyl (C)]': 'C',
            'M[Oxidation (M)]': '1',
            'S[Phospho (STY)]': '2',
            'T[Phospho (STY)]': '3',
            'Y[Phospho (STY)]': '4',
        }

        def trans_str_mod_to_int(pep):
            pep = pep.replace('_', '')
            for mod, int_mod in str_mod_to_int_dict.items():
                pep = pep.replace(mod, int_mod)
            return pep

        self.result_df['IntSeq'] = self.result_df['EG.ModifiedPeptide'].apply(trans_str_mod_to_int)

    def add_is_phos(self):
        self.result_df['IsPhospep'] = self.result_df['EG.PTMPositions [Phospho (STY)]'].apply(lambda x: True if pd.notna(x) else False)

    def add_phossite_in_pep(self, ptm_thres=0.99):
        def _add_pep_site(x, p_t):
            if not x['IsPhospep']:
                return np.nan
            prob_list = x['EG.PTMProbabilities [Phospho (STY)]'].split(';')
            site = [s for i, s in enumerate(x['EG.PTMPositions [Phospho (STY)]'].split(';')) if float(prob_list[i]) >= p_t]
            return ';'.join(site)

        self.result_df['PepPhosPos'] = self.result_df.apply(_add_pep_site, args=(ptm_thres,), axis=1)

    def add_phossite_in_prot(self):
        def _add_prot_site(x):
            if pd.isna(x['PepPhosPos']):
                return np.nan
            pep_pos = int(x['PEP.PeptidePosition'].split(';')[0].split(',')[0])
            phos_site_in_prot = [int(_) + pep_pos - 1 for _ in x['PepPhosPos'].split(';')]
            return ';'.join(map(str, phos_site_in_prot))

        self.result_df['ProtPhosPos'] = self.result_df.apply(_add_prot_site, axis=1)

    def add_protein_phossite_comb(self):
        def _add_comb_site(x):
            if pd.isna(x['ProtPhosPos']):
                return np.nan
            else:
                return ';'.join([x['FirstProtein'] + ',' + _ for _ in x['ProtPhosPos'].split(';')])
        self.result_df['ProtPhosPosComb'] = self.result_df.apply(_add_comb_site, axis=1)

    def add_predefined_features(self, feature_list=None, **kwargs):
        """
        :param feature_list TODO 最后应该把所有feature单独添加，这里传一个list来选择添加哪些feature
        """
        self.add_first_protein()
        self.add_prec()
        self.add_intseq()
        self.add_is_phos()
        self.add_phossite_in_pep(ptm_thres=kwargs['ptm_thres'])
        self.add_phossite_in_prot()
        self.add_protein_phossite_comb()

    def count_loss_type(self):
        return self.result_df['F.FrgLossType'].value_counts()

    def __getitem__(self, item):
        return self.result_df[item]

    def __setitem__(self, key, value):
        self.result_df[key] = value

    def get_columns(self):
        return self.result_df.columns

    def set_columns(self, cols):
        if not isinstance(cols, list):
            cols = [cols]
        self.result_df.columns = cols

    columns = property(get_columns, set_columns, doc='''Columns of result df''')

    def get_frag_inten(self) -> dict:
        result_spec = dict()
        reps = self.result_df['R.Replicate'].drop_duplicates().tolist()  # 这里应该增加可以选择 Replicate 或是 FileName
        try:
            t = tqdm(reps)
            for rep in t:
                one_rep_result = self.result_df[self.result_df['R.Replicate'] == rep]
                one_rep_spec_dict = dict()
                for prec, df in one_rep_result.groupby('FG.Id'):  # TODO FG.Id 不在 result 列中
                    frag_dict = dict()
                    for row_index, row in df.iterrows():
                        frag_dict[f'{row["F.FrgIon"]}+{row["F.Charge"]}-{row["F.FrgLossType"]}'] = row['F.NormalizedPeakArea']
                    one_rep_spec_dict[prec] = frag_dict
                result_spec[rep] = one_rep_spec_dict
        except Exception:
            t.close()
            raise
        return result_spec

    def recalc_exp_irt(self):
        self.result_df['ReCalciRT'] = self.result_df.apply(
            lambda x: (x['EG.MeanApexRT'] - x['EG.StartRT']) / (
                    x['EG.EndRT'] - x['EG.StartRT']) * (
                    x['EG.EndiRT'] - x['EG.StartiRT']) + x['EG.StartiRT'], axis=1)

    def return_result_df(self, original_cols=False):
        return self.result_df


class SpectronautResultPivotPeptide(SpectronautResult):
    """
    TODO
    1. 读取search result
    2. 对search result加入protein list分类
    3. 提取相同的rep（给定ident分为对应的result）
    """
    def __init__(self):
        super(SpectronautResultPivotPeptide, self).__init__()


class SpectronautResultPivotProtein(SpectronautResult):
    def __init__(self):
        super(SpectronautResultPivotProtein, self).__init__()


# TODO: Re-write this


def get_one_prefix_result(result_df, prefix, suffixes):
    return [set(result_df[f'{prefix}-{each_suffix}'].dropna()) for each_suffix in suffixes]


def get_search_result(result_file, sheet_names, prefixes=('Protein', 'Precursor'), suffixes=('R1', 'R2', 'R3')):
    if isinstance(sheet_names, str):
        sheet_names = [sheet_names]
    with pd.ExcelFile(result_file) as f:
        result_list = []
        for sheet in sheet_names:
            df = f.parse(sheet_name=sheet)
            store_dict = dict()
            for each_prefix in prefixes:
                store_dict[each_prefix] = get_one_prefix_result(df, each_prefix, suffixes)
                store_dict[f'{each_prefix}-Total'] = rapid_kit.data_struc_kit.sum_set_in_list(
                    store_dict[each_prefix])
            if 'Precursor' in prefixes:
                store_dict['Peptide'] = [set([_.split(
                    '.')[0] for _ in each_suff_data]) for each_suff_data in store_dict[each_prefix]]
                store_dict['Peptide-Total'] = rapid_kit.data_struc_kit.sum_set_in_list(
                    store_dict['Peptide'])
            result_list.append(store_dict)
    return result_list


def select_target_df(original_df, region_identifier):
    region_df = rapid_kit.extract_df_with_col_ident(original_df, region_identifier, focus_col='R.Instrument (parsed from filename)')
    return region_df


def read_search_result_intensity(result_file):
    result_df = pd.read_csv(result_file, sep='\t', low_memory=False)
    region_intensity_list = []
    for _ in ['region3', 'region5', 'region6']:
        region_intensity_dict = dict()
        each_region_df = select_target_df(result_df, _)
        for each_prec in each_region_df['EG.PrecursorId'].drop_duplicates():
            each_prec_df = each_region_df[each_region_df['EG.PrecursorId'] == each_prec]
            noloss_prec_df = each_prec_df[each_prec_df['F.FrgLossType'] == 'noloss']
            fragment_list = (noloss_prec_df['F.FrgType'] +
                             noloss_prec_df['F.FrgNum'].astype(str) + '+' +
                             noloss_prec_df['F.Charge'].astype(str)).tolist()
            fragment_intensity = noloss_prec_df['F.MeasuredRelativeIntensity'].tolist(
            )
            region_intensity_dict[each_prec] = dict(
                zip(fragment_list, fragment_intensity))
        region_intensity_list.append(region_intensity_dict)
    return region_intensity_list
