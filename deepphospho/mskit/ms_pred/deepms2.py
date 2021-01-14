import re

from mskit.rapid_kit import substring_finder


def extract_mod_for_deepms2(mod_pep, mod_trans_dict: dict):
    """
    From SN modpep to DeepMS2 modification col needed format
    For mods:
        Acetyl is not considered
        Carbamidomethyl will be ignored
    """
    mod_pep = mod_pep.replace('_', '')
    if '[' not in mod_pep:
        return ''
    else:
        strip_pep, mod_pos, mods = substring_finder(mod_pep)

        mod_info = []
        for pos, mod in zip(mod_pos, mods):
            if 'Carbamidomethyl' in mod:
                continue
            mod_aa = strip_pep[pos - 1]
            mod_name = mod_trans_dict[mod]
            mod_info.append(f'{mod_aa}{pos}{mod_name}')
        return ';'.join(mod_info)


def read_deepms2_phos_result(file, aa_mod_trans_dict: dict):
    """
    From DeepMS2-Phospho to spectra dict with key-value list {int_prec: {spec_dict}, ...}
    Use the .msp file processed by the R script but not the raw .json pred result

    columns = ['StrippedPeptide', 'PrecCharge', 'IntPrec', 'ModInfo', 'IntenDict']
    df =  pd.DataFrame(results, columns=columns)
    df['IntenDict'] = df['IntenDict'].apply(
        lambda x: {frag: inten for frag, inten in x.items() if re.findall('[by](\d+)\+', frag)[0] not in ['1', '2']})
    df['IntenDict'] = df['IntenDict'].apply(mskit.calc.normalize_intensity, max_num=100)
    """
    pred_results = []
    one_spec = dict()
    with open(file, 'r') as f:
        for row in f:
            row = row.strip('\n')

            if row.startswith('Name'):
                pep, prec_charge = row.replace('Name: ', '').split('/')
            elif row.startswith('Comment'):
                mods = row.split('Mods=')[1]
                if mods == '0':
                    mod_pep = pep
                else:
                    split_mods = [mod.split(',') for mod in mods.split('/')][1:]
                    split_mods = [mod for mod in split_mods if mod[2] != 'Carbamidomethyl']
                    if len(split_mods) == 0:
                        mod_pep = pep
                    else:
                        split_mods = [[int(x[0]), *x[1:]] for x in split_mods]
                        split_mods = sorted(split_mods, key=lambda x: x[0])
                        split_pep_seq = list(pep)

                        for mod_idx, mod_aa, mod_name in split_mods:
                            split_pep_seq[mod_idx] = aa_mod_trans_dict[mod_name][mod_aa]
                        mod_pep = ''.join(split_pep_seq)
                int_prec = f'@{mod_pep}.{prec_charge}'
            elif row.startswith('Num peaks'):
                pass
            elif not row:
                if one_spec:
                    pred_results.append([pep, prec_charge, int_prec, mods, one_spec.copy()])
                    one_spec = dict()
                else:
                    pass
            else:
                frag_mz, frag_inten, frag_name = row.split('\t')
                frag_type, frag_num, frag_lossnum, frag_charge = re.findall(r'([by])(\d+?)(-\d+?)?\^(\d)', frag_name)[0]
                if frag_lossnum == '':
                    frag_losstype = 'Noloss'
                elif frag_lossnum == '-17':
                    frag_losstype = 'NH3'
                elif frag_lossnum == '-18':
                    frag_losstype = 'H2O'
                elif frag_lossnum == '-98':
                    frag_losstype = '1,H3PO4'
                else:
                    raise
                trans_frag_name = f'{frag_type}{frag_num}+{frag_charge}-{frag_losstype}'
                one_spec[trans_frag_name] = float(frag_inten)
    return pred_results
