import operator
import os
import pickle
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import ipdb
from deepphospho.configs import config_main as cfg

from deepphospho.utils.utils_functions import match_frag, ion_types, get_index, get_pkl_path, intensity_load_check
import logging
import math

PADDING_CHAR = '#'
ENDING_CHAR = '$'
MASK_CHAR = "-"
CLS_TOKEN = "&"

ALPHABET = {
    PADDING_CHAR: 0,
    "A": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "K": 9,
    "L": 10,
    "M": 11,
    "N": 12,
    "P": 13,
    "Q": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "V": 18,
    "W": 19,
    "Y": 20,
    "1": 21,
}

GBSC = {
    PADDING_CHAR: 0,
    ENDING_CHAR: 0,
    "A": 0,
    "C": 0,
    "D": 784,
    "E": 790,
    "F": 0,
    "G": 0,
    "H": 927.84,
    "I": 0,
    "K": 926.74,
    "L": 0,
    "M": 830,
    "N": 864.94,
    "P": 0,
    "Q": 865.25,
    "R": 1000,
    "S": 775,
    "T": 780,
    "V": 0,
    "W": 909.53,
    "Y": 790,
    "1": 830,
    "2": 775,
    "3": 780,
    "4": 790
}

Scaled_GBSC = {k: v / 1000 for k, v in GBSC.items()}
# "*" represents the Acetyl modification, "@" represents no


class Dictionary(object):
    """
    from the characters to the integer idx
    """

    def __init__(self, path):
        logger = logging.getLogger("IonIntensity")

        self.word2idx = {}
        self.idx2word = []
        # for padding char:
        self.add_word(PADDING_CHAR)
        self.add_word(ENDING_CHAR)
        self.add_word(MASK_CHAR)
        self.add_word(CLS_TOKEN)

        self.build(path)

        # some unknown items..
        self.add_word("U")
        self.add_word("X")

        # print out the dictionary to see:
        # print(self.__str__())
        logger.info(self.__str__())

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def build(self, path):
        assert os.path.exists(path)
        # construct the dictionary:
        # load all data characters and unique them for index
        print("building up Dictionary")
        seq_data = pd.read_json(path)
        pep = [i.split(".")[0] for i in seq_data.columns.values]
        for aa in sorted(set(''.join(pep))):
            self.add_word(aa)

    def __len__(self):
        return len(self.idx2word)

    def __str__(self):
        res_str = '>> number of aa: %d \n' % len(self.idx2word)
        for aa in sorted(self.word2idx.items(), key=operator.itemgetter(1)):
            res_str += str(aa[1]) + '->' + str(aa[0]) + '\n'
        return res_str


class IonData(object):
    """
    Tokenize the sequences
    """

    def __init__(self, data_cfg, path=None, dictionary=None, ):
        """

        :param data_cfg:
        :param path: data path
        :param dictionary: use exists dictionary to tokenize
        """
        logger = logging.getLogger("IonIntensity")
        if path is None:
            path = os.path.join(data_cfg['data_path'], data_cfg['data_fn'])
        assert os.path.exists(path)
        if dictionary is None:
            dictionary = Dictionary(path)

        self.dictionary = dictionary

        # load from the cache
        if data_cfg['refresh_cache']:
            to_delete_path = get_pkl_path(path)
            if os.path.exists(to_delete_path):
                os.remove(to_delete_path)

        else:
            pre_load_data_path = get_pkl_path(path)

            if os.path.exists(pre_load_data_path):
                logger.info(f"Use use {pre_load_data_path}!")
                # print(f"Use use {pre_load_data_path}!")
                with open(pre_load_data_path, 'rb') as f:
                    pre_load_data = pickle.load(f)
                self.X1 = pre_load_data["X1"]
                self.X2 = pre_load_data["X2"]
                self.y = pre_load_data['y']
                if cfg.TRAINING_HYPER_PARAM['remove_ac_pep']:  # here to remove peptide of ac in N terminal after save cache
                    ac_token = self.dictionary.word2idx["*"]
                    pep_index_without_ac = self.X1[:, 0] != ac_token
                    total_sample = self.X1.shape[0]
                    no_ac_sample = np.sum(pep_index_without_ac)
                    self.X1 = self.X1[pep_index_without_ac, ]
                    self.X2 = self.X2[pep_index_without_ac, ]
                    self.y = self.y[pep_index_without_ac, ]

                    logger.info(f'Remove {total_sample - no_ac_sample} ac_pep from total {total_sample} peps ')
                # ipdb.set_trace()
                return

        logger.info(f"Reading Files...\n{path}")
        if path.split(".")[-1] == 'json':
            seq_data = pd.read_json(path)
            # the expected data format: rows and columns are ion types and precursors, correspondingly
        else:
            seq_data = pd.read_csv(path)
        if 'To_Predict' in data_cfg and data_cfg['To_Predict']:
            pep_info = seq_data[data_cfg['SEQUENCE_FIELD_NAME']]
        else:
            pep_info = seq_data.columns.values
        if cfg.DATA_PROCESS_CFG['MAX_SEQ_LEN'] is None:
            count_max = lambda pep_list: max([len(aas.split(".")[0])-1 for aas in pep_list])
            # here get the len of each peptide in input and minus one is as a result of ac modification
            self.MAX_SEQ_LEN = count_max(pep_info)
        else:
            self.MAX_SEQ_LEN = cfg.DATA_PROCESS_CFG['MAX_SEQ_LEN']
        N_seq = len(pep_info)
        self.number_seq = N_seq
        self.N_aa = len(dictionary)
        data_size = N_seq
        self.data_size = data_size

        if cfg.TRAINING_HYPER_PARAM['DEBUG']:
            data_size = 2000
        self.X1 = np.zeros((data_size, self.MAX_SEQ_LEN + 2))
        self.X2 = np.zeros((data_size, self.MAX_SEQ_LEN + 2))

        if cfg.TRAINING_HYPER_PARAM['pdeep2mode']:
            ion_type_len = 8
        elif cfg.TRAINING_HYPER_PARAM['only_two_ions']:
            ion_type_len = 2
        else:
            ion_type_len = 12

        self.y = np.zeros((data_size, self.MAX_SEQ_LEN + 2, ion_type_len))

        # todo add multiprocessing here?
        for seq_index, seq in tqdm(enumerate(pep_info), total=len(pep_info)):
            # fill in X:
            if cfg.TRAINING_HYPER_PARAM['DEBUG']:
                if seq_index >= data_size:
                    print("debug on, break")
                    break

            pep = seq.split(".")[0]
            aa_length = len(pep) - 1  # because we add "*" to represent the Acetyl modification and "@" to represent no
            wrapped_seq = pep + ENDING_CHAR
            charge = int(seq.split(".")[1])

            self.X1[seq_index, :len(wrapped_seq)] = [self.dictionary.word2idx[aa] for aa in wrapped_seq]
            self.X2[seq_index, :len(wrapped_seq)] = [charge for aa in wrapped_seq]

            if not data_cfg['To_Predict']:

                raw_intensity = seq_data.iloc[:, seq_index][seq_data.iloc[:, seq_index] > 0]
                to_load_intensity = dict(raw_intensity / np.max(raw_intensity))

                # if seq_index == 16706:
                #     ipdb.set_trace()
                for i in range(aa_length):
                    i += 1
                    for j in range(ion_type_len):
                        matches = False
                        matches = match_frag(ion_types(aa_length, i)[j],
                                             to_load_intensity)
                        if matches != False:
                            match_key = matches[1]
                            # import ipdb
                            # ipdb.set_trace()
                            self.y[seq_index][i][j] = to_load_intensity[match_key]

                intensity_load_check(cfg, to_load_intensity, self.y[seq_index])
            if cfg.TRAINING_HYPER_PARAM['add_phos_principle']:
                if len(get_index(pep, '2', '3')) != 0:  # for loss one phos
                    min_phos_index = min(get_index(pep, '2', '3'))
                    max_phos_index = max(get_index(pep, '2', '3'))
                    if cfg.TRAINING_HYPER_PARAM['pdeep2mode']:
                        self.y[seq_index][:min_phos_index, 2:4] = -2
                        self.y[seq_index][max_phos_index:, 6:] = -2
                        # here we remove +1 as same row represents bn, yn-1, n is the length of amino acids

                    else:

                        self.y[seq_index][:min_phos_index, 2] = -2
                        self.y[seq_index][max_phos_index:, 8] = -2
                        # here we remove +1 as same row represents bn, yn-1, n is the length of amino acids

                else:
                    if cfg.TRAINING_HYPER_PARAM['pdeep2mode']:
                        self.y[seq_index][:, 2:4] = -2
                        self.y[seq_index][:, 6:] = -2
                    else:
                        self.y[seq_index][:, 2] = -2
                        self.y[seq_index][:, 8] = -2
                if not cfg.TRAINING_HYPER_PARAM['pdeep2mode']:  # for loss two phos

                    if len(get_index(pep, '2', '3')) < 2:

                        self.y[seq_index][:, 3] = -2
                        self.y[seq_index][:, 9] = -2
                    else:
                        phos_index = sorted(get_index(pep, '2', '3'))
                        self.y[seq_index][:phos_index[1], 3] = -2
                        self.y[seq_index][phos_index[-2]:, 9] = -2
                        # here we remove +1 as same row represents bn, yn-1, n is the length of amino acids

            self.y[seq_index][0][:] = -1  # No fragment of complete length (b0,y_aa_len), we set -1
            self.y[seq_index][aa_length:][:] = -1  # No fragment of y0, b_aa_len and longer than precursor, set -1

        if not cfg.TRAINING_HYPER_PARAM['DEBUG'] and not data_cfg['To_Predict']:
            pkl_path = get_pkl_path(path)
            with open(pkl_path, 'wb') as f:
                pickle.dump({'X1': self.X1, 'X2': self.X2, 'y': self.y}, f)

        if cfg.TRAINING_HYPER_PARAM['remove_ac_pep']:  # here to remove peptide of ac in N terminal after saving cache

            ac_token = self.dictionary.word2idx["*"]
            pep_index_without_ac = self.X1[:, 0] != ac_token
            total_sample = self.X1.shape[0]
            no_ac_sample = np.sum(pep_index_without_ac)
            self.X1 = self.X1[pep_index_without_ac, ]
            self.X2 = self.X2[pep_index_without_ac, ]
            self.y = self.y[pep_index_without_ac, ]
            logger.info(f'Remove {total_sample - no_ac_sample} ac_pep from total {total_sample} peps ')

        logger.info(f"Reading Files...Done! {path}")


class RTdata(object):

    """
    Tokenize the sequences
    """

    def __init__(self, data_cfg, path=None, dictionary=None, ):
        """
        :param data_cfg:
        :param path: data path
        :param dictionary: use exists dictionary to tokenize
        """
        if path is None:
            path = os.path.join(data_cfg['data_path'], data_cfg['data_fn'])
        assert os.path.exists(path)
        if dictionary is None:
            dictionary = Dictionary(path)
        logger = logging.getLogger("RT")
        self.dictionary = dictionary
        _, file_extension = os.path.splitext(path)
        # ipdb.set_trace()
        logger.info(f"Reading Files...! \n{path}")
        if file_extension == '.csv':
            seq_data = pd.read_csv(path)
        else:
            seq_data = pd.read_csv(path, sep="\t")
        SEQUENCE_FIELD_NAME = data_cfg['SEQUENCE_FIELD_NAME']
        RT_FIELD_NAME = data_cfg['RT_FIELD_NAME']
        # ipdb.set_trace()

        if cfg.DATA_PROCESS_CFG['MAX_SEQ_LEN'] is None:
            count_max = lambda pep_list: max([len(aas)-1 for aas in pep_list])
            # here get the len of each peptide in input and minus one is as a result of ac modification
            self.MAX_SEQ_LEN = count_max(seq_data[SEQUENCE_FIELD_NAME])
        else:
            self.MAX_SEQ_LEN = cfg.DATA_PROCESS_CFG['MAX_SEQ_LEN']
        if RT_FIELD_NAME not in seq_data.columns.values:
            seq_data[RT_FIELD_NAME] = np.zeros(len(seq_data[SEQUENCE_FIELD_NAME]))

        if cfg.DATA_PROCESS_CFG['MIN_RT'] is None:
            self.MIN_RT = math.floor(min(seq_data[RT_FIELD_NAME]))
        else:
            self.MIN_RT = cfg.DATA_PROCESS_CFG['MIN_RT']

        if cfg.DATA_PROCESS_CFG['MAX_RT'] is None:
            self.MAX_RT = math.ceil(max(seq_data[RT_FIELD_NAME]))
        else:
            self.MAX_RT = cfg.DATA_PROCESS_CFG['MAX_RT']

        N_seq = len(seq_data[SEQUENCE_FIELD_NAME])
        self.number_seq = N_seq
        self.N_time_step = self.MAX_SEQ_LEN + 2  # ending and ac tokens
        self.N_aa = len(dictionary)
        data_size = N_seq
        if cfg.TRAINING_HYPER_PARAM['DEBUG']:
            data_size = 50000
        self.scale_by_zero_one_on = data_cfg['SCALE_BY_ZERO_ONE']

        # load from the cache
        if os.path.exists(path + '.pkl'):
            if data_cfg['refresh_cache']:
                os.remove(path + '.pkl')
            elif cfg.Use_cache:
                with open(path + '.pkl', 'rb') as f:
                    pre_load_data = pickle.load(f)

                self.X1 = pre_load_data["X1"][:data_size]
                self.y = pre_load_data['y'][:data_size]
                # ipdb.set_trace()
                # if cfg.TRAINING_HYPER_PARAM['add_hydro']:
                #     self.X2 = pre_load_data["X2"][:data_size]
                # if cfg.TRAINING_HYPER_PARAM['add_rc']:
                #     self.X3 = pre_load_data["X3"][:data_size]
                return
        self.X1 = np.zeros((data_size, self.MAX_SEQ_LEN + 2))
        self.y = np.zeros(data_size)

        # todo add multiprocessing here?
        for seq_index, seq in tqdm(enumerate(seq_data[SEQUENCE_FIELD_NAME]), total=len(seq_data[SEQUENCE_FIELD_NAME])):
            if cfg.TRAINING_HYPER_PARAM['DEBUG']:
                if seq_index >= data_size:
                    print("debug on, break")
                    break
            if seq_data.iloc[seq_index][RT_FIELD_NAME] > self.MAX_RT or seq_data.iloc[seq_index][RT_FIELD_NAME] < self.MIN_RT:
                continue

            wrapped_seq = seq + ENDING_CHAR

            try:
                self.X1[seq_index, :len(wrapped_seq)] = [self.dictionary.word2idx[aa] for aa in wrapped_seq]
            except ValueError:
                ipdb.set_trace()
            if 'To_Predict' in data_cfg and data_cfg['To_Predict']:
                continue
            else:
                self.y[seq_index] = seq_data.iloc[seq_index][RT_FIELD_NAME]

        logger.info("Y max: %f min: %f" % (self.MAX_RT, self.MIN_RT))
        # ipdb.set_trace()
        if data_cfg["SCALE_BY_ZERO_ONE"]:
            # Normalize by even distribution
            self.y = (self.y - self.MIN_RT) / (self.MAX_RT - self.MIN_RT)
        # ipdb.set_trace()
        if not cfg.TRAINING_HYPER_PARAM['DEBUG'] and cfg.Save_cache:
            with open(path + '.pkl', 'wb') as f:
                pickle.dump({'X1': self.X1, 'y': self.y}, f)
        logger.info(f"Reading Files...Done! {path}")


class Detectdata(object):

    def __init__(self, data_cfg, path=None, dictionary=None, ):
        if path is None:
            path = os.path.join(data_cfg['data_path'], data_cfg['data_fn'])
        assert os.path.exists(path)
        if dictionary is None:
            dictionary = Dictionary(path)
        logger = logging.getLogger("Detect")
        self.dictionary = dictionary
        _, file_extension = os.path.splitext(path)
        logger.info(f"Reading Files...{path}")
        if file_extension == '.csv':
            seq_data = pd.read_csv(path)
        else:
            seq_data = pd.read_csv(path, sep="\t")

        SEQUENCE_FIELD_NAME = data_cfg['SEQUENCE_FIELD_NAME']
        Detect = data_cfg['Detect']
        self.MAX_SEQ_LEN = cfg.DATA_PROCESS_CFG['MAX_SEQ_LEN']
        N_seq = len(seq_data[SEQUENCE_FIELD_NAME])
        self.number_seq = N_seq
        self.N_aa = len(dictionary)
        data_size = N_seq
        if cfg.TRAINING_HYPER_PARAM['DEBUG']:
            data_size = 1000

        # load from the cache
        if os.path.exists(path + '.pkl'):
            if data_cfg['refresh_cache']:
                os.remove(path + '.pkl')
            elif cfg.Use_cache:
                with open(path + '.pkl', 'rb') as f:
                    pre_load_data = pickle.load(f)

                self.X1 = pre_load_data["X1"][:data_size]
                self.y = pre_load_data['y'][:data_size]
                # ipdb.set_trace()
                # if cfg.TRAINING_HYPER_PARAM['add_hydro']:
                #     self.X2 = pre_load_data["X2"][:data_size]
                # if cfg.TRAINING_HYPER_PARAM['add_rc']:
                #     self.X3 = pre_load_data["X3"][:data_size]
                return
        self.X1 = np.zeros((data_size, self.MAX_SEQ_LEN + 1))
        self.y = np.zeros(data_size)

        for seq_index, seq in tqdm(enumerate(seq_data[SEQUENCE_FIELD_NAME]), total=len(seq_data[SEQUENCE_FIELD_NAME])):
            if cfg.TRAINING_HYPER_PARAM['DEBUG']:
                if seq_index >= data_size:
                    print("debug on, break")
                    break

            wrapped_seq = seq + ENDING_CHAR

            try:
                self.X1[seq_index, :len(wrapped_seq)] = [self.dictionary.word2idx[aa] for aa in wrapped_seq]
            except KeyError:
                ipdb.set_trace()
            if 'To_Predict' in data_cfg and data_cfg['To_Predict']:
                continue
            else:
                self.y[seq_index] = seq_data.iloc[seq_index][Detect]

        if not cfg.TRAINING_HYPER_PARAM['DEBUG'] and cfg.Save_cache:
            with open(path + '.pkl', 'wb') as f:
                pickle.dump({'X1': self.X1, 'y': self.y}, f)
        logger.info(f'>> Read Detect dataset done; source:{path}')


