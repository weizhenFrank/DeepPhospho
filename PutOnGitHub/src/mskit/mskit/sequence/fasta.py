from .ted import TED

import os
import re
import numpy as np

from mskit import rapid_kit


class FastaWriter(object):
    pass


def ktx_to_dict(input_file, keystarter='<'):
    """ parsing keyed text to a python dictionary. """
    answer = dict()

    with open(input_file, 'r+', encoding='utf-8') as f:
        lines = f.readlines()

    k, val = '', ''
    for line in lines:
        if line.startswith(keystarter):
            k = line.replace(keystarter, '').strip()
            val = ''
        else:
            val += line

        if k:
            answer.update({k: val.strip()})

    return answer


class FastaParser(object):
    """
    TODO 传入path，或content，或handle，增加skiprow和commend ident
    TODO 两个 fasta parser 合并
    """
    def __init__(self, fasta_path, parse_rule='uniprot', preprocess=True, nothing_when_init=False):
        """
        :param fasta_path:
        :param parse_rule: 'uniprot' will get the second string for title split by '|', and others will be the first string split by get_one_prefix_result blank,
        while maybe other formats of fasta title is needed later
        """

        if os.path.exists(fasta_path):
            self.fasta_path = os.path.abspath(fasta_path)
        else:
            print('Incorrect fasta file path')
            raise FileNotFoundError(f'The Fasta is not existed: {fasta_path}')

        self.parse_rule = parse_rule
        self.id_parse_func = None
        self.comment_parse_func = None

        self.raw_title_dict = dict()
        self.prot_acc_dict = dict()

        self.raw_content = None  # The whole text of the fasta file
        self._protein_info = dict()  # The description information of each protein in the fasta file
        self._protein_to_seq = dict()  # The whole sequence of each protein (No digestion)
        self._seq_to_protein = dict()  # Digested peptide to protein. The protein may be str if one else list.
        self._seq_list = []  # Digested peptides of all protein sequence in the fasta file

        self.fasta_file_stream = None

        if preprocess:
            self.init_fasta()

    def get_parse_rule(self):
        return self.parse_rule

    def set_parse_rule(self, parse_rule=None, id_parse_func=None, comment_parse_func=None):
        id_parse_rules = {'uniprot': lambda x: re.findall('^>(.+?)\|(.+?)\|(.+?)$', x)}
        comment_parse_rules = {'uniprot': lambda x: None}
        if isinstance(parse_rule, str):
            try:
                self.id_parse_func = id_parse_rules[parse_rule.lower()]  # TODO 设置一个 parse rule dict
            except KeyError:
                raise KeyError(f'Not {parse_rule} Found in The Predefined rule list')
        else:
            pass
        self.parse_rule = parse_rule

        if id_parse_func:
            self.id_parse_func = id_parse_func
        if comment_parse_func:
            self.comment_parse_func = comment_parse_func

    parse_rule = property(get_parse_rule, set_parse_rule, doc='''Return type can be seq or site_seq.
    If seq: A list of seq will be returned. ['ADEFHK', 'PQEDAK' , ...]
    If site_seq: A list of site and seq will be returned. [(0, 'ADEFHK'), (12, 'PQEDAK'), ...]''')

    def __call__(self, *args, **kwargs):
        pass

    def __iter__(self):
        return iter(self.get_total_seqlist())

    def __getitem__(self, item):
        return self.prot_acc_dict[item]

    def __setitem__(self, key, value):
        self.raw_title_dict[key] = value

    def __enter__(self):
        self.fasta_file_stream = open(self.fasta_path, 'r')

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fasta_file_stream.close()

    def __add__(self, other):  # 只保留唯一 title
        pass

    @staticmethod
    def merge_fasta(f1, f2, unique_title=False, unique_seq=False):
        pass

    def get_raw_content(self):
        """
        Get the whole content of the input fasta file
        """
        if not self.raw_content:
            with open(self.fasta_path, 'r') as fasta_handle:
                self.raw_content = fasta_handle.read()
        return self.raw_content

    def get_all_title(self):
        pass

    def init_fasta(self, method='re'):
        if not self.raw_content:
            self.get_raw_content()
        if method == 're':
            title_seq_list = re.findall('(>.+?\\n)([^>]+\\n?)', self.raw_content)
            title_seq_list = [(title.strip('\n'), seq.replace('\n', '')) for title, seq in title_seq_list]
        else:
            raise
        self.raw_title_dict = dict(title_seq_list)

    def add_new_seqs(self, new_seq_dict, id_conflict=None):
        """
        id_conflict:
            new: Keep new protein seq
            origin: Keep original protein seq
            go_on: Add as PROTEIN-2
            consistent: Add as PROTEIN-2 and rename the original key to PROTEIN-1
        """
        self.raw_title_dict.update(new_seq_dict)

    def to_file(self, file_path, seq_line=None):
        """
        seq_line:
            None: Write one seq to one line
            80: Write one seq to multilines to keep the numebr of char in one line == 80
            other integer: Keep number of char equal to the customed number
        """
        with open(file_path, 'w') as f:
            if seq_line:
                for title, seq in self.raw_title_dict.items():
                    f.write(title + '\n')
                    f.write(''.join([seq[_ * seq_line: (_ + 1) * seq_line] + '\n' for _ in range(int(np.ceil(len(seq) / seq_line)))]))
            else:
                for title, seq in self.raw_title_dict:
                    f.write(title + '\n')
                    f.write(seq + '\n')

    def one_protein_generator(self):
        """
        Generate title and sequence of each protein in fasta file
        """
        seq_title = ''
        seq_list = []
        with open(self.fasta_path, 'r') as fasta_handle:
            for _line in fasta_handle:
                if not _line:
                    print('Blank line existed in fasta file')  # TODO 记录 blank line 的行号
                    continue
                if _line.startswith('>'):
                    if seq_title and seq_list:
                        yield seq_title, ''.join(seq_list)
                    seq_title = _line.strip('\n')
                    seq_list = []
                else:
                    seq_list.append(_line.strip('\n'))
            if seq_title and seq_list:
                yield seq_title, ''.join(seq_list)

    def protein2seq(self, protein_info=False):
        if not self._protein_to_seq:
            for _title, _seq in self.one_protein_generator():
                protein_ident = rapid_kit.fasta_title(_title, self.parse_rule)
                self._protein_to_seq[protein_ident] = _seq
                if protein_info:
                    self._protein_info[protein_ident] = _title
        return self._protein_to_seq

    def seq2protein(
            self,
            miss_cleavage=(0, 1, 2),
            min_len=7,
            max_len=33) -> dict:

        if not self._seq_to_protein:
            if not self._protein_to_seq:
                self.protein2seq()

            ted = TED(miss_cleavage=miss_cleavage,
                      min_len=min_len,
                      max_len=max_len,
                      enzyme='Trypsin',
                      return_type='seq')
            for protein_acc, seq in self._protein_to_seq.items():
                compliant_seq = ted(seq)
                for _each_seq in compliant_seq:
                    self._seq_list.append(_each_seq)
                    if _each_seq not in self._seq_to_protein:
                        self._seq_to_protein[_each_seq] = protein_acc
                    else:
                        if isinstance(self._seq_to_protein[_each_seq], str):
                            self._seq_to_protein[_each_seq] = [
                                self._seq_to_protein[_each_seq], protein_acc]
                        elif isinstance(self._seq_to_protein[_each_seq], list):
                            self._seq_to_protein[_each_seq].append(
                                protein_acc)
        return self._seq_to_protein

    def get_total_seqlist(
            self,
            miss_cleavage=(0, 1, 2),
            min_len=7,
            max_len=33):

        if not self._seq_list:
            self.seq2protein(miss_cleavage=miss_cleavage,
                             min_len=min_len,
                             max_len=max_len)
        self._seq_list = rapid_kit.drop_list_duplicates(self._seq_list)
        return self._seq_list


class _FastaParser(FastaParser, ):
    def __init__(self, fasta_type='protein'):
        if fasta_type.lower() == 'protein':
            super(FastaParser, self).__init__()
        elif fasta_type.lower() == 'base' or fasta_type.lower() == 'nucleic acid':
            pass
