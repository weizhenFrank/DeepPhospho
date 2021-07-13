import os
import re
import json

import pandas as pd

from deep_phospho import proteomics_utils as prot_utils
from deep_phospho.proteomics_utils.post_analysis import spectronaut as SN


def generate_spec_lib(data_name, output_folder, pred_ion_path, pred_rt_path, save_path=None, logger=None):
    if logger is not None:
        logger.info(f'Reading predicted results for {data_name}')
    with open(pred_ion_path, 'r') as f:
        pred_ion = json.load(f)
    pred_rts = dict(pd.read_csv(pred_rt_path, sep='\t')[['sequence', 'pred']].values)

    pred_lib_rows = []
    loss_num = 0
    if logger is not None:
        logger.info(f'Start generating library for {data_name}')
    for intprec, pred_spec in pred_ion.items():
        if 'U' in intprec or 'X' in intprec:
            continue
        intpep, prec_charge = intprec.split('.')
        modpep = SN.sn_utils.intseq_to_sn_modpep(intpep)
        strip_pep = re.sub(r'\[.+?\]', '', modpep.replace('_', ''))

        if intpep in pred_rts:
            pred_rt = pred_rts[intpep]
        else:
            loss_num += 1
            continue

        prec = f'{modpep}.{prec_charge}'
        prec_mz = prot_utils.calc.calc_prec_mz(prec)
        prec_basic_data_list = [prec_charge, modpep, strip_pep, pred_rt, modpep, prec_mz]

        pred_spec = prot_utils.calc.normalize_intensity(pred_spec, max_num=100)
        pred_spec = prot_utils.calc.keep_top_n_inten(pred_spec, top_n=30)

        for frag, inten in pred_spec.items():
            frag_type, frag_num, frag_charge, frag_losstype = re.findall(r'([abcxyz])(\d+)\+(\d)-(.+)', frag)[0]
            if int(frag_num) in (1, 2):
                continue
            if float(inten) <= 5:
                continue
            frag_mz = prot_utils.calc.calc_fragment_mz(modpep, frag_type, frag_num, frag_charge, frag_losstype)
            frag_losstype = SN.sn_constant.LossType.Readable_to_SN[frag_losstype]
            pred_lib_rows.append(prec_basic_data_list + [frag_losstype, frag_num, frag_type, frag_charge, frag_mz, inten])

    pred_lib_df = pd.DataFrame(pred_lib_rows, columns=SN.SpectronautLibrary.LibBasicCols)

    pred_lib_df['Prec'] = pred_lib_df['ModifiedPeptide'] + '.' + pred_lib_df['PrecursorCharge'].astype(str)
    if logger is not None:
        logger.info(f'Total {len(set(pred_lib_df["Prec"]))} precursors in initial {data_name} library')

    pred_lib_df = pred_lib_df.groupby('Prec').filter(lambda x: len(x) >= 3)
    if logger is not None:
        logger.info(f'Total {len(set(pred_lib_df["Prec"]))} precursors in final {data_name} library')

    pred_lib_df = pred_lib_df[SN.SpectronautLibrary.LibBasicCols]

    if save_path is not None:
        lib_path = save_path
    else:
        lib_path = os.path.join(output_folder, f'Library-{data_name}-DP_I5_n30.xls')
    if logger is not None:
        logger.info(f'Saving generated library to {lib_path}')
    pred_lib_df.to_csv(lib_path, sep='\t', index=False)
    return lib_path


def merge_lib(main_lib_path, add_libs_path, output_folder, task_name, save_path=None, logger=None):
    if logger is not None:
        logger.info(f'Loading main library {main_lib_path}')
    main_lib = SN.SpectronautLibrary(main_lib_path)
    main_lib.add_intpep()
    main_lib = main_lib.to_df()

    # TODO add_libs_path can be either list or dict
    for add_lib_name, add_lib_path in add_libs_path.items():
        if logger is not None:
            logger.info(f'Loading additional library {add_lib_path}')
        add_lib = SN.SpectronautLibrary(add_lib_path)
        add_lib.retain_basic_cols()
        add_lib.add_intpep()
        add_lib = add_lib.to_df()

        added_peps = set(add_lib['IntPep']) - set(main_lib['IntPep'])
        add_lib = add_lib[add_lib['IntPep'].isin(added_peps)]

        add_lib['Prec'] = add_lib['ModifiedPeptide'] + '.' + add_lib['PrecursorCharge'].astype(str)
        add_lib = add_lib.groupby('Prec').filter(lambda x: len(x) >= 3)
        add_lib = add_lib[SN.SpectronautLibrary.LibBasicCols + ['IntPep']]

        main_lib = main_lib.append(add_lib)

    main_lib = main_lib[SN.SpectronautLibrary.LibBasicCols]

    if save_path is None:
        save_path = os.path.join(output_folder, f'HybridLibrary-{task_name}-DP_I5_n30.xls')
    if logger is not None:
        logger.info(f'Saving hybrid library {save_path}')
    main_lib.to_csv(save_path, sep='\t', index=False)
    return save_path
