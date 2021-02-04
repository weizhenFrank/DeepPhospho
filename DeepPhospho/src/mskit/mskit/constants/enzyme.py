"""
In class Enzyme

Enzymes:
    The dict contains all usable enzymes
    Key: The name of enzyme, which starts with uppercase letter except some conventions
    Values:
        CleavageSite: a tuple with three elements, ('cleavage aa', 'direction', 'exclusion')
            'cleavage aa': The AA for cleavage, one AA with one letter representation
            'direction': N-term or C-term of the cleavage AA, -1 and 1
            'exclusion': The AA that makes enzyme miss the target AA
                This will have three situation: None, one AA, and multi AA in one string (None, 'P', 'EPQ')

digestion_site:
    Get AA pairs of the input enzyme.
    For example:
        digestion_site('Trypsin') -> ['KA', 'KC', 'KD', ..., 'RY'] with no 'KP' or 'RP' in it

"""
from .aa import AA


class Enzyme:
    Enzymes = {
        'Trypsin': {
            'CleavageSite': [('K', 1, 'P'), ('R', 1, 'P')],
        },
        'LysC': {
            'CleavageSite': [('K', 1, 'P')],
        },
        'LysN': {
            'CleavageSite': [('K', -1, None)],
        },
        'AspC': {
            'CleavageSite': [('D', 1, None)],
        },
        'AspN': {
            'CleavageSite': [('D', -1, None)],
        },
        'Chymotrypsin': {
            'CleavageSite': [('F', 1, None), ('W', 1, None), ('Y', 1, None)],
        },
    }

    def digestion_site(self, enzyme):
        e = self.Enzymes[enzyme]
        site_info = e['CleavageSite']
        aa_pair = []
        for _info in site_info:
            target_aa = _info[0]
            _direction = _info[1]
            exclusion = _info[2]
            if exclusion:
                accompanied_aa = [_ for _ in AA.AAList_20 if _ not in exclusion]
            else:
                accompanied_aa = AA.AAList_20
            for _acc in accompanied_aa:
                if _direction == 1:
                    _pair = f'{target_aa}{_acc}'
                elif _direction == -1:
                    _pair = f'{_acc}{target_aa}'
                else:
                    raise
                aa_pair.append(_pair)
        return aa_pair

    def __call__(self, enzyme):
        return self.digestion_site(enzyme)
