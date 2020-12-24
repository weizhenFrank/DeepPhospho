"""
主要包含以下内容
stripped序列
修饰序列（默认）
修饰信息（str）
RT

包含以下方法
长度
修饰数
可变修饰数（total number或 number of each mod in dict）
固定修饰数
aa统计
TED
remove mod
add mod（随机或指定位点）

"""


class Peptide(object):
    def __init__(self, *args):
        self.name = None
        self.score = None  # This may be a dict if any score is stored
        self.add_info = None  # May be a dict to store some additional information of this peptide

    def get_mod(self):
        pass

    def mod_num(self):
        pass

    def get_var_mod(self):
        pass

    def var_mod_num(self):
        pass

    def set_mod_prop(self, site=None, mod_num=None, prop=None):
        pass

    def filter_mod_prop(self):
        pass

    def get_fix_mod(self):
        pass

    def fix_mod_num(self):
        pass

    def pep_len(self):
        pass

    def __len__(self):
        pass

    def ted(self, *args):
        pass

    def get_mc(self, cleavage_aa='KR'):
        pass

    def get_mass(self):
        pass

    def set_exp_mass(self):
        pass


class PeptideGroups(object):
    pass
