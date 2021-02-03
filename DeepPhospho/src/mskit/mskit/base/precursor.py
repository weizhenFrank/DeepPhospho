"""
继承自peptide，增加
charge
intensity info
"""

from . import peptide


class Precursor(peptide.Peptide):
    def __init__(self, ):
        super(Precursor, self).__init__()
        self.frags = None

    def get_frag_list(self, ion_type='by'):
        pass

    def get_charge(self):
        pass

    def set_charge(self):
        pass

    charge = property(get_charge, set_charge, doc='''''')

    def get_fragments(self):
        return self.frags

    def set_fragments(self, frag, update: bool = False):
        if isinstance(frag, dict):
            if update:
                pass
            else:
                pass
        elif isinstance(frag, (tuple, list)):
            pass
        else:
            try:
                pass
            except TypeError:
                raise TypeError('')

    frag = property(get_fragments, set_fragments, doc='''''')

    def get_inten_by_index(self, frag_type, frag_num):
        pass

    def get_mz(self):
        pass

    def set_exp_mz(self):
        pass
