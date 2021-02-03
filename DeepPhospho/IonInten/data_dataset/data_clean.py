import numpy as np
import pandas as pd
import re
# for new data format adaptation
# import ipdb
import os


def old_operations():

    intensity = pd.read_json("/Users/admin/Downloads/20200714-Trainset-Inten-PhosDIA_18-DIALib_91_seed0_Normed_train_no_holdout_666.json")

    # All_mods = []
    # for i in intensity['Mod info']:
    #     All_mods += re.split('[, ;]', i)
    #     # print(type(i))
    # # set(All_mods)
    #
    #
    # Modpep = []
    # for i in intensity['Modpep']:
    #     Modpep.append(
    #         i.replace('_', '').replace('M(ox)', '1').replace('S(ph)', '2').replace('T(ph)', '3').replace('Y(ph)', '4'))

    # Modpep
    #
    # for i in Modpep:
    #     if 'ph' in i:
    #         print(i)
    # Modpep = intensity.index.values
    # Modpep = [i.split(".")[0] for i in Modpep]
    # ipdb.set_trace()
    # for i in intensity['Modpep']:
    #     Modpep.append(
    #         i.replace('_', '').replace('M(ox)', '1').replace('S(ph)', '2').replace('T(ph)', '3').replace('Y(ph)', '4'))
    loss = []
    for j in intensity['normalized_intensity']:
        for i in j.keys():
            if '+' in i:
                loss.append(i.split('+')[0][0]+i.split('+')[1])
    loss = set(loss)
    # ipdb.set_trace()
    # print(loss)
    # for i in loss:
    #     print(i, '\n')
    # float_str = []
    # for j in intensity['Frag inten']:
    #     for v in j.values():
    #         # print(type(v) == str)
    #         float_str.append(type(v) == str)


        # for v in j.values():
        #     print(type(v))
        #     float()

    # print(float_str)
    # print(float_str)
    # print(all(float_str))
    # Frag_inten = []
    # for j in intensity['Frag inten']:
    #     for k, v in j.items():
    #         # print(type(v) == str)
    #         j[k] = float(v)
    #     Frag_inten.append(j)
    #
    # frag_relative_inten = []
    # Frag_inten = intensity['Intensity']
    # for j in Frag_inten:
    #     m = dict()
    #     for k, v in j.items():
    #         # print(type(v) == str)
    #         m[k] = v/max(j.values())
    #     frag_relative_inten.append(m)
    # ipdb.set_trace()


    # print(Frag_inten)

    # float_str = []
    # for j in Frag_inten:
    #     for v in j.values():
    #         # print(type(v) == str)
    #         float_str.append(type(v) == float)

    # print(all(float_str))
    # Info = pd.DataFrame({'sequence': Modpep, 'charge': intensity['Charge'], 'normalized_intensity': frag_relative_inten, 'intensity': Frag_inten})
    # Info.to_csv("/Users/admin/Downloads/20200622-Inten-Jeff-MaxScore.txt", index=None, sep='\t')
    # Info.to_json("/Users/admin/Downloads/202007-Inten-Jeff-MaxScore.json")
    # Info = pd.DataFrame({'sequence': Modpep, 'charge': intensity['PrecCharge'], 'normalized_intensity': frag_relative_inten, 'intensity': Frag_inten})
    # Info.to_csv("/Users/admin/Downloads/20200731-PhosDIA-DDAUniquePhosprec-NoAc_Normed.txt", index=None, sep='\t')
    # Info.to_json("/Users/admin/Downloads/20200731-PhosDIA-DDAUniquePhosprec-NoAc_Normed.json")
    # Info = pd.DataFrame({'sequence': Modpep, 'charge': intensity['PrecCharge'], 'normalized_intensity': frag_relative_inten, 'intensity': Frag_inten})
    # Info.to_csv("/Users/admin/Downloads/20200714-Trainset-Inten-PhosDIA_18-DIALib_91_seed0_Normed.txt", index=None, sep='\t')
    # Info.to_json("/Users/admin/Downloads/20200714-Trainset-Inten-PhosDIA_18-DIALib_91_seed0_Normed.json")

# Ion = pd.read_csv("/Users/admin/Downloads/20200622-Inten-Jeff-MaxScore.txt")
# Ion = pd.read_csv("/Users/admin/Downloads/20200622-Inten-Jeff-MaxScore.txt")

def get_outpath(path, append_name):
    output_dir = os.path.dirname(path)
    out_file_name = f"{os.path.splitext(os.path.basename(path))[0]}_{append_name}.csv"
    out_path = os.path.join(output_dir, out_file_name)
    return out_path


def remove_redundant(path):
    pep = pd.read_csv(path)
    redundant = set(pep['sequence'][pep['detect'] == 0]) & set(pep['sequence'][pep['detect'] == 1])
    filtered_pep = pep[~(pep['sequence'].isin(list(redundant)) & (pep['detect'] == 0))]
    output_dir = os.path.dirname(path)
    out_file_name = f"{os.path.splitext(os.path.basename(path))[0]}_filtered.csv"
    out_path = os.path.join(output_dir, out_file_name)
    filtered_pep.to_csv(out_path, index=True, index_label=False)


def remove_unkown_token(token, path):
    pep_filtered = pd.read_csv(path)
    removed = [token not in i for i in pep_filtered['sequence']]
    out_path = get_outpath(path, "remove_unknown_token")
    pep_filtered[removed].to_csv(out_path, index=True, index_label=False)


def main():
    # remove_redundant("/Users/admin/Documents/Project/Detectability/Detect/Detectability_all.csv")
    remove_unkown_token("8", "/Users/admin/Documents/Project/Detectability/Detect/Detectability_all_filtered.csv")


if __name__ == '__main__':
    main()





