import copy
import ipdb

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess_input_data import MASK_CHAR, PADDING_CHAR, CLS_TOKEN, ENDING_CHAR
from deepphospho.configs import config_main as cfg


class RandomMaskingDataset(Dataset):
    def __init__(self, ion_data, de_mod=True, mask_modifier=True,
                 mask_ratio=0.15):
        self.ion_data = ion_data
        self.mask_modifier = mask_modifier
        self.de_mod = de_mod
        if mask_modifier or de_mod:
            self.mask_ratio = 1
        else:
            self.mask_ratio = mask_ratio

        self.char_num = len(self.ion_data.dictionary.word2idx)

    def __getitem__(self, item):

        x = np.array(self.ion_data.X1[item], dtype=np.long).squeeze()
        y = copy.deepcopy(x)
        if self.de_mod:
            mod_no_mod_idx = \
                np.nonzero(np.logical_or(np.logical_or(np.logical_or(x == self.ion_data.dictionary.word2idx['1'],
                                                                     x == self.ion_data.dictionary.word2idx['2']),
                                                       np.logical_or(x == self.ion_data.dictionary.word2idx['3'],
                                                                     x == self.ion_data.dictionary.word2idx['4'])
                                                       ),
                                         np.logical_or(np.logical_or(x == self.ion_data.dictionary.word2idx['M'],
                                                                     x == self.ion_data.dictionary.word2idx['S']),
                                                       np.logical_or(x == self.ion_data.dictionary.word2idx['T'],
                                                                     x == self.ion_data.dictionary.word2idx['Y'])
                                                       ))

                           )[0]
            select_idx = np.nonzero(np.logical_or(np.logical_or(x == self.ion_data.dictionary.word2idx['1'],
                                                                x == self.ion_data.dictionary.word2idx['2']),
                                                  np.logical_or(x == self.ion_data.dictionary.word2idx['3'],
                                                                x == self.ion_data.dictionary.word2idx['4'])
                                                  ))[0]
        else:
            # selecting to-replace index
            select_idx = np.nonzero(np.logical_and(np.logical_and(x != self.ion_data.dictionary.word2idx[PADDING_CHAR],
                                                                  x != self.ion_data.dictionary.word2idx[CLS_TOKEN]),
                                                   x != self.ion_data.dictionary.word2idx[ENDING_CHAR]))[0]

        np.random.shuffle(select_idx)
        mask_num = int(np.ceil(len(select_idx) * self.mask_ratio))
        mask_idx = select_idx[:mask_num]
        # ipdb.set_trace()
        if mask_num != 0:
            # replace_idx = mask_idx[: int(len(mask_idx) * 0.1)]
            # mask_idx = mask_idx[int(len(mask_idx) * 0.1): ]

            # There is a bug when the mask_num=0 as x[mask_idx]=x where all x=self.rt_data.dictionary.word2idx[MASK_CHAR]
            # do masking and replacing
            if self.de_mod:
                for i in mask_idx:
                    if x[i] == self.ion_data.dictionary.word2idx['1']:
                        x[i] = self.ion_data.dictionary.word2idx['M']
                        # print('change 1-->M')
                    elif x[i] == self.ion_data.dictionary.word2idx['2']:
                        x[i] = self.ion_data.dictionary.word2idx['S']
                        # print('change 2-->S')
                    elif x[i] == self.ion_data.dictionary.word2idx['3']:
                        x[i] = self.ion_data.dictionary.word2idx['T']
                        # print('change 3-->T')
                    else:
                        x[i] = self.ion_data.dictionary.word2idx['Y']
                        # print('change 4-->Y')
                loss_acc_idx = np.zeros_like(x, dtype=bool)
                loss_acc_idx[mod_no_mod_idx] = 1

            else:
                x[mask_idx] = self.ion_data.dictionary.word2idx[MASK_CHAR]
                # np.random.shuffle(self.char_list)
                # x[replace_idx] = self.char_list[:len(replace_idx)]

                # mark the masked index for calculating loss
                loss_acc_idx = np.zeros_like(x, dtype=bool)
                loss_acc_idx[mask_idx] = 1

        else:
            loss_acc_idx = np.zeros_like(x, dtype=bool)
        # print('#'*10)
        # print('this is x', x)
        # print('-'*10)
        # print('this is y', y)
        # print('@' * 10)
        # print('this is loss_acc_idx', loss_acc_idx)
        return x, (y, loss_acc_idx)

    def __len__(self):
        return len(self.ion_data.X1)


class IonDataset(Dataset):
    def __init__(self, ion_data):
        self.ion_data = ion_data

    def __getitem__(self, item):

        x1 = np.array(self.ion_data.X1[item], dtype=np.long).squeeze()
        y = np.array(self.ion_data.y[item], dtype=np.float)
        if hasattr(self.ion_data, "X2"):
            x2 = np.array(self.ion_data.X2[item], dtype=np.float).squeeze()
            x2 = x2.reshape(-1, 1)

            if cfg.data_name == 'Prosit':
                x3 = np.array(self.ion_data.X3[item], dtype=np.float).squeeze()
                x3 = x3.reshape(-1, 1)

                return x1, x2, x3, y
            else:
                return x1, x2, y
        else:
            return x1, y

    def __len__(self):
        return len(self.ion_data.X1)


def collate_fn(batch):
    if cfg.TRAINING_HYPER_PARAM["Bert_pretrain"]:
        transposed_batch = list(zip(*batch))
        X = np.stack(transposed_batch[0])
        # contains intial tokens and masked tokens flags
        supervisions = np.stack(transposed_batch[1])
        Y = np.stack([supervisions[:, 0], supervisions[:, 1, ]])
        Y = torch.from_numpy(Y).long()

        return torch.from_numpy(X), Y

    else:

        batch = list(zip(*batch))
        if len(batch) == 4:
            X1 = np.stack(batch[0])
            X2 = np.stack(batch[1])
            X3 = np.stack(batch[2])
            Y = np.stack(batch[3])
            # Y = torch.from_numpy(Y).float()
            # ipdb.set_trace()

            return (torch.from_numpy(X1).long(), torch.from_numpy(X2).float(),
                    torch.from_numpy(X3).float()), torch.from_numpy(
                Y).float()

        elif len(batch) == 3:
            X1 = np.stack(batch[0])
            X2 = np.stack(batch[1])
            Y = np.stack(batch[2])

        # Y = torch.from_numpy(Y).float()
        # ipdb.set_trace()
            return (torch.from_numpy(X1).long(), torch.from_numpy(X2).float()), torch.from_numpy(Y).float()
        else:
            X1 = np.stack(batch[0])
            Y = np.stack(batch[1])
            return torch.from_numpy(X1).long(), torch.from_numpy(Y).float()


