from collections import defaultdict
import matplotlib.pyplot as plt

import pandas as pd


pFind = 3


class PFindConst:
    ProteinSummaryComment = '----Summary----'
    ProteinCutoffComment = '-' * 40


class ProteinFile(object):
    """
    REV_sp
    ...
    REV_CON
    """
    def __init__(self, file_path):
        self.file_path = file_path

        self.main_title = None
        self.sub_title = None

        self.protein_group_title = ['']

        self.protein_list = list()
        self.prec_list = list()
        self.prec_list = list()

    def _parse_protein_file(self):
        with open(self.file_path, 'r') as f:
            self.main_title = f.readline().strip('\n').split('\t')
            self.sub_title = f.readline().strip('\n\t').split('\t')
            for row in f:
                strip_row = row.strip('\n\t')
                if row == PFindConst.ProteinCutoffComment:
                    pass

    def pfind_protein_df(self, p, remove_rev=True, remove_con=False):
        protein_main = []
        protein_sub = []
        with open(p, 'r') as f:
            main_title = f.readline().strip('\n').split('\t')
            f.readline()
            for row in f:
                if '-' * 40 in row:
                    continue
                if '----Summary' in row:
                    break
                split_row = row.strip('\n').split('\t')
                if split_row[0].isdigit():
                    current_protein = split_row[1]
                    main_score = split_row[2]
                    main_q = split_row[3]
                    protein_main.append(split_row[:-1])
                else:
                    if split_row[1]:
                        protein_sub.append([current_protein, main_score, main_q, split_row[1], split_row[2], split_row[3]])
        protein_main_df = pd.DataFrame(protein_main, columns=main_title)
        protein_sub_df = pd.DataFrame(protein_sub, columns=['MainProtein', 'MainScore', 'MainQ-Value', 'Type', 'Protein', 'Q-Value'])
        if remove_rev:
            protein_main_df = protein_main_df[~protein_main_df['AC'].str.contains('REV_')]
            protein_sub_df = protein_sub_df[~protein_sub_df['Protein'].str.contains('REV_')]
        if remove_con:
            protein_main_df = protein_main_df[~protein_main_df['AC'].str.contains('CON_')]
            protein_sub_df = protein_sub_df[~protein_sub_df['Protein'].str.contains('CON_')]
        return protein_main_df, protein_sub_df

    def get_protein_list(self):
        Protein_MainDF, Protein_SubDF = self.pfind_protein_df(PATH_pFindResults['Test'], remove_rev=True, remove_con=True)
        ProtMain = set(Protein_MainDF[Protein_MainDF['Q-Value'].astype(float) < 0.01]['AC'])
        ProtMain_ACC = [_.split('|')[1] for _ in ProtMain]
        ProtTotal = set(Protein_MainDF['AC']) | set(Protein_SubDF['Protein'])
        ProtTotal_ACC = [_.split('|')[1] for _ in ProtTotal]
        print(len(ProtMain_ACC))
        print(len(ProtTotal_ACC))

    def plot_protein_num(self):
        plt.figure(figsize=(2.5, 1.), dpi=300)
        for i, protein_list in enumerate([ProtMain_ACC, ProtTotal_ACC]):
            plt.bar(i, len(protein_list), )
            plt.annotate(format(len(protein_list), ','), (i, len(protein_list) / 2), va='center', ha='center')
        bpk.drawing_area.set_thousand_separate()
        plt.xticks(list(range(2)), ['Main', 'Total'], rotation=45)
        plt.title('Total proteins in results')


class SpecFile(object):
    def __init__(self):
        pass

    @staticmethod
    def pfind_spec_prot_to_file(x):  # 把这个函数放在外面
            proteins = x['Proteins'].strip('/').split('/')
            file = x['File_Name'].split('.')[0]
            prot_len = len(proteins)
            return dict(zip(proteins, [file] * prot_len))

    def get_prot_to_file_dict(self):
        prot_to_file_list = SpecDF.apply(self.pfind_spec_prot_to_file, axis=1).tolist()
        prot_to_file = defaultdict(set)
        for pro_file in prot_to_file_list:
            for name, file in pro_file.items():
                if name.startswith('REV_') or name.startswith('CON_'):
                    continue
                for r_index, region_ident in enumerate(RegionIdent):
                    if region_ident in file:
                        prot_to_file[name.split('|')[1]].add(f'R{r_index+1}')
        print('Total prot -> file dict (contains REV_):', len(prot_to_file.keys()))
        return prot_to_file
