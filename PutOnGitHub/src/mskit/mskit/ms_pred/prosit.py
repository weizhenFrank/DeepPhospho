import os
from mskit.inherited_builtins import NonOverwriteDict


def read_prosit_fragment_result(prosit_result):
    """
    '[Oxidation (M)]'
    '[Carbamidomethyl (C)]'
    """
    with open(os.path.abspath(prosit_result), 'r') as prosit_handle:
        prosit_title = prosit_handle.readline()
        title_dict = dict([(__, _) for _, __ in enumerate(prosit_title.strip('\n').split(','))])
        predicted_fragment_data = dict()

        for each_line in prosit_handle:
            if not each_line or each_line == '\n':
                continue
            split_line = each_line.strip('\n').split(',')
            if split_line[0] == '0':
                continue

            loss_type = split_line[title_dict['FragmentLossType']]
            if loss_type != 'noloss':
                continue
            mod_pep = split_line[title_dict['ModifiedPeptide']]
            charge = split_line[title_dict['PrecursorCharge']]
            intensity = split_line[title_dict['RelativeIntensity']]
            fragment_num = split_line[title_dict['FragmentNumber']]
            fragment_type = split_line[title_dict['FragmentType']]
            fragment_charge = split_line[title_dict['FragmentCharge']]

            prec = charge + mod_pep
            fragment_name = '{}{}+{}'.format(fragment_type, fragment_num, fragment_charge)
            if prec not in predicted_fragment_data:
                predicted_fragment_data[prec] = {fragment_name: float(intensity)}
            else:
                predicted_fragment_data[prec][fragment_name] = float(intensity)
    return predicted_fragment_data


def read_prosit_irt_result(prosit_result):
    with open(os.path.abspath(prosit_result), 'r') as prosit_handle:
        prosit_title = prosit_handle.readline()
        title_dict = dict([(__, _) for _, __ in enumerate(prosit_title.strip('\n').split(','))])
        predicted_irt_data = NonOverwriteDict()

        for each_line in prosit_handle:
            if not each_line or each_line == '\n':
                continue
            split_line = each_line.strip('\n').split(',')
            if split_line[0] == '0':
                continue

            irt = split_line[title_dict['iRT']]
            mod_pep = split_line[title_dict['ModifiedPeptide']]

            int_replaced_pep = mod_pep.strip('_').replace('C[Carbamidomethyl (C)]', 'C').replace('M[Oxidation (M)]', '1')
            if '[' in int_replaced_pep:
                continue
            predicted_irt_data[int_replaced_pep] = float(irt)
    return predicted_irt_data


def spectronaut_to_prosit_testset(test_lib_path, test_set_output):
    with open(test_lib_path, 'r') as handle_lib, open(test_set_output, 'w') as handle_test:
        lib_title = handle_lib.readline()
        title_dict = dict([(__, _) for _, __ in enumerate(lib_title.strip('\n').split('\t'))])
        handle_test.write('modified_sequence,collision_energy,precursor_charge\n')

        temp_prec_list = []
        for each_lib_line in handle_lib:
            split_line = each_lib_line.strip('\n').split('\t')
            stripped_pep = split_line[title_dict['StrippedPeptide']]
            if len(stripped_pep) > 30:
                continue

            mod_pep = split_line[title_dict['ModifiedPeptide']]
            charge = split_line[title_dict['PrecursorCharge']]

            current_prec = charge + mod_pep
            if current_prec in temp_prec_list:
                continue
            mod_pep = mod_pep.replace('[Carbamidomethyl (C)]', '')
            mod_pep = mod_pep.replace('[Oxidation (M)]', '(ox)')
            mod_pep = mod_pep.replace('_', '')
            if '[' in mod_pep:
                continue
            handle_test.write('{},{},{}\n'.format(mod_pep, '30', charge))
            temp_prec_list.append(current_prec)


def pdeep_input_to_prosit(pdeep_input_path, prosit_input):
    with open(pdeep_input_path, 'r') as handle_pdeep_input, open(prosit_input, 'w') as handle_prosit:
        pdeep_input_title = handle_pdeep_input.readline()
        title_dict = dict([(__, _) for _, __ in enumerate(pdeep_input_title.strip('\n').split('\t'))])
        handle_prosit.write('modified_sequence,collision_energy,precursor_charge\n')

        temp_prec_list = []
        for each_predict_input_line in handle_pdeep_input:
            split_line = each_predict_input_line.strip('\n').split('\t')
            stripped_pep = split_line[title_dict['peptide']]
            if len(stripped_pep) > 30:
                continue

            charge = split_line[title_dict['charge']]

            mod_info = split_line[title_dict['modification']]
            current_prec = charge + stripped_pep + mod_info
            if current_prec in temp_prec_list:
                continue

            oxi_mod = []
            if mod_info:
                mod_info = mod_info.split(';')[:-1]
                for each_mod in mod_info:
                    if 'Oxidation[M]' in each_mod:
                        oxi_mod.append(int(each_mod.split(',')[0]))
                if oxi_mod:
                    mod_pep = ''
                    previous_oxi_site = 0
                    for each_oxi in oxi_mod:
                        mod_pep += stripped_pep[previous_oxi_site: each_oxi] + '(ox)'
                        previous_oxi_site = each_oxi
                    mod_pep += stripped_pep[previous_oxi_site:]
                    pep = mod_pep
                else:
                    pep = stripped_pep
            else:
                pep = stripped_pep

            handle_prosit.write('{},{},{}\n'.format(pep, '30', charge))
            temp_prec_list.append(current_prec)
