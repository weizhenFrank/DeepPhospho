import re


class TED(object):
    def __init__(self, miss_cleavage=(0, 1, 2), min_len=7, max_len=33, enzyme='Trypsin', return_type='seq'):
        """
        Theoretical Enzyme Digestion -> TED
        :param miss_cleavage: this can be int or tuple, while it will be converted into tuple when use
        :param min_len:
        :param max_len:
        :param enzyme: only trypsin is supported now
        :param return_type: 'seq' or 'site_seq'
        """
        self._mc = miss_cleavage
        self._min_len = min_len
        self._max_len = max_len
        self._enzyme = enzyme
        self._return_type = return_type

    def get_mc(self):
        return self._mc

    def set_mc(self, mc):
        if isinstance(mc, tuple):
            pass
        elif isinstance(mc, int):
            mc = (mc, )
        else:
            try:
                mc = (int(mc), )
            except TypeError:
                raise TypeError('Miss cleavage shoule be int or tuple of int')
        self._mc = mc

    mc = property(get_mc, set_mc, doc='''Miss cleavage for enzyme digestion''')

    def get_enzyme(self):
        return self._enzyme

    def set_enzyme(self, enzyme):
        if not isinstance(enzyme, str):
            if isinstance(enzyme, (list, tuple)):
                enzyme = enzyme[0]
                Warning(f'The setted enzyme is not str, now get the first value of that: {enzyme}')
            else:
                raise TypeError(f'The input of enzyme should be str, now {type(enzyme)}')
        self._enzyme = enzyme

    enzyme = property(get_enzyme, set_enzyme, doc='''Enzyme used for digestion''')

    def get_return_type(self):
        return self._return_type

    def set_return_type(self, return_type: str):
        if return_type not in ['seq', 'site_seq']:
            raise TypeError(f'The input of return_type should be \'seq\' or \'site_seq\', now: {return_type}')
        self._return_type = return_type

    return_type = property(get_return_type, set_return_type, doc='''Return type can be seq or site_seq.
    If seq: A list of seq will be returned. ['ADEFHK', 'PQEDAK' , ...]
    If site_seq: A list of site and seq will be returned. [(0, 'ADEFHK'), (12, 'PQEDAK'), ...]''')

    def __call__(self, seq):
        seq = seq.replace('\n', '').replace(' ', '')

        split_seq_list = [(_.start(), _.group())
                          for _ in re.finditer('.*?[KR](?!P)|.+', seq)]

        compliant_seq = []
        for i in range(len(split_seq_list)):
            for mc in self._mc:
                one_seq = ''.join([_[1] for _ in split_seq_list[i: i + mc + 1]])
                if self._min_len <= len(one_seq) <= self._max_len:
                    if self._return_type == 'seq':
                        compliant_seq.append(one_seq)
                    elif self._return_type == 'site_seq':
                        compliant_seq.append((split_seq_list[i][0], one_seq))
        return compliant_seq

