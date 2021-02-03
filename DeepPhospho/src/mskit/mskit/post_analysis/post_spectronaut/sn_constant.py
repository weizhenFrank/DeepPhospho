"""
1.
This stores many different modification display methods, and all the modification will be got from here.

Shorthand:
Spectronaut -> SN

"""


class LossType(object):
    SN_to_Readable = {
        'noloss': 'Noloss',
        'H3PO4': '1,H3PO4',
        'H2O': '1,H2O',
        '1(+H2+O)1(+H3+O4+P)': '1,H2O;1,H3PO4',
        '2(+H3+O4+P)': '2,H3PO4',
        'NH3': '1,NH3',
        '1(+H3+N)1(+H3+O4+P)': '1,NH3;1,H3PO4',
        '1(+H2+O)2(+H3+O4+P)': '1,H2O;2,H3PO4',
        '1(+H9+N+O8+P2)': '1,NH3;2,H3PO4',
        '3(+H3+O4+P)': '3,H3PO4',
        '1(+H2+O)3(+H3+O4+P)': '1,H2O;3,H3PO4',
        '1(+H3+N)3(+H3+O4+P)': '1,NH3;3,H3PO4',
        '4(+H3+O4+P)': '4,H3PO4',
        '1(+H2+O)4(+H3+O4+P)': '1,H2O;4,H3PO4',
    }

    Readable_to_SN = {v: k for k, v in SN_to_Readable.items()}


class BasicModInfo(object):
    ModDict_new = {'C': 'C[Carbamidomethyl (C)]',
                   'M': 'M[Oxidation (M)]'}
    ModDict_old = {'C': 'C[Carbamidomethyl]',
                   'M': 'M[Oxidation]'}

    ModConvert_new = {}
    ModConvert_old = {}

    DeepRTIntModDict_new = {
        'C[Carbamidomethyl (C)]': 'C',
        'M[Oxidation (M)]': '1',
        'S': '2',
        'T': '3',
        'Y': '4'}
    DeepRTIntModDict_old = {
        'C[Carbamidomethyl]': 'C',
        'M[Oxidation]': '1',
        'S': '2',
        'T': '3',
        'Y': '4'}
    # IntModDict = {'M[+16]': '1', 'S[+80]': '2', 'T[+80]': '3', 'Y[+80]': '4'}

    pDeepModType = {'C': 'Carbamidomethyl[C]', 'M': 'Oxidation[M]'}


class ModType(BasicModInfo):
    """
    Spectronaut version 12 has get_one_prefix_result different modification display type.
    The default version is set to 12, which uses the new modification display method.
    The version should be set in each main functions but not the functions that are used frequently.
    """
    def __init__(self, spectronaut_version=12):
        self._spectronaut_version = spectronaut_version
        self.ModDict = self.ModDict_new
        self.DeepRTIntModDict = self.DeepRTIntModDict_new

        self.pDeepMod2SNMod_new = dict(
            [(self.pDeepModType[aa], self.ModDict_new[aa][1:]) for aa in self.pDeepModType])
        self.pDeepMod2SNMod_old = dict(
            [(self.pDeepModType[aa], self.ModDict_old[aa][1:]) for aa in self.pDeepModType])
        self.pDeepMod2SNMod = self.pDeepMod2SNMod_new

    def set_spectronaut_version(self, version):
        self._spectronaut_version = version
        if self._spectronaut_version >= 12:
            self.ModDict = self.ModDict_new
            self.DeepRTIntModDict = self.DeepRTIntModDict_new
            self.pDeepMod2SNMod = self.pDeepMod2SNMod_new
        else:
            self.ModDict = self.ModDict_old
            self.DeepRTIntModDict = self.DeepRTIntModDict_old
            self.pDeepMod2SNMod = self.pDeepMod2SNMod_old

    def get_spectronaut_int_version(self):
        return self._spectronaut_version

    def get_spectronaut_str_version(self):
        if self._spectronaut_version >= 12:
            return 'new'
        else:
            return 'old'

    @staticmethod
    def get_mod_dict(ver='new'):
        if isinstance(ver, str):
            if ver == 'new':
                return ModType.ModDict_new
            elif ver == 'old':
                return ModType.ModDict_old
            else:
                raise NameError('Choose mod version from \'new\' and \'old\'')
        elif isinstance(ver, int):
            if ver >= 12:
                return ModType.ModDict_new
            else:
                return ModType.ModDict_old


class SNLibraryTitle(object):
    LibraryMainCol = [
        'PrecursorCharge',
        'ModifiedPeptide',
        'StrippedPeptide',
        'iRT',
        'LabeledPeptide',
        'PrecursorMz',
        'FragmentLossType',
        'FragmentNumber',
        'FragmentType',
        'FragmentCharge',
        'FragmentMz',
        'RelativeIntensity',
        'ProteinGroups']
    LibraryMainColPGOut = [
        'PrecursorCharge',
        'ModifiedPeptide',
        'StrippedPeptide',
        'iRT',
        'LabeledPeptide',
        'PrecursorMz',
        'FragmentLossType',
        'FragmentNumber',
        'FragmentType',
        'FragmentCharge',
        'FragmentMz',
        'RelativeIntensity']
