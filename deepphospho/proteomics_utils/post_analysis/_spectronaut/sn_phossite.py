import numpy as np
import pandas as pd

from deepphospho.proteomics_utils.rapid_kit import substring_finder
from .sn_utils import sn_modpep_to_intseq


def init_phos_analysis(df,
                       protein_col='PG.ProteinGroups',
                       modpep_col='EG.ModifiedPeptide',
                       prec_charge_col='FG.Charge',
                       ):
    df['FirstProtein'] = df[protein_col].apply(lambda x: x.split(';')[0])
    df['Precursor'] = df[modpep_col] + '.' + df[prec_charge_col].astype(int).astype(str)
    df['IntPep'] = df[modpep_col].apply(sn_modpep_to_intseq)
    df['IntPrec'] = df['IntPep'] + '.' + df[prec_charge_col].astype(str)

    df['PhosNum'] = df[modpep_col].apply(lambda x: x.count('[Phospho (STY)]'))
    df['IsPhospep'] = df['PhosNum'].apply(lambda x: True if x > 0 else False)
    df['IsMod'] = df[modpep_col].apply(lambda x: True if '[' in x else False)
    df['DropPhosPep'] = df[modpep_col].apply(lambda x: x.replace('[Phospho (STY)]', ''))
    return df


def add_phossite_in_pep(x, modpep_col='EG.ModifiedPeptide',
                        prob_thres=None,
                        prob_col='EG.PTMProbabilities [Phospho (STY)]',
                        modpos_col='EG.PTMPositions [Phospho (STY)]',
                        required_mod='[Phospho (STY)]'):
    if not x['IsPhospep']:
        return np.nan

    strippep, modpos, mods = substring_finder(x[modpep_col].replace('_', ''))
    required_mod_pos = [pos for pos, mod in zip(modpos, mods) if mod == required_mod]

    if prob_thres:
        prob_list = list(map(float, x[prob_col].split(';')))
        peppos_list = list(map(int, x[modpos_col].split(';')))
        thres_filter_pos = [peppos
                            for prob, peppos
                            in zip(prob_list, peppos_list)
                            if peppos in required_mod_pos and prob >= prob_thres]
    else:
        thres_filter_pos = required_mod_pos

    if thres_filter_pos:
        if len(thres_filter_pos) == len(required_mod_pos):
            return ';'.join(list(map(str, thres_filter_pos)))
        else:
            return 'SiteFilter'
    else:
        return 'PepFilter'


def add_phossite_in_prot(x, ):
    if pd.isna(x['PepPhosPos']):
        return np.nan

    peppos_in_prot = str(x['PEP.PeptidePosition']).split(';')
    prot_mod_site = ''
    pep_mod_sites = x['PepPhosPos'].split(';')

    for protein, peppos in zip(x['PG.ProteinGroups'].split(';'), peppos_in_prot):
        if peppos == '':
            prot_mod_site += ';'
            continue
        for one_pep_pos in peppos.split(','):
            for one_mod_site in pep_mod_sites:
                prot_mod_site += f'{int(one_pep_pos) + int(one_mod_site) - 1},'
            prot_mod_site = prot_mod_site.strip(',')
            prot_mod_site += '/'
        prot_mod_site = prot_mod_site.strip('/')
        prot_mod_site += ';'
    prot_mod_site = prot_mod_site[:-1]

    return prot_mod_site

