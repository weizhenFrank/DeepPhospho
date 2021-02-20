def element_aa2res(aa_comp: dict) -> dict:
    res_comp = aa_comp.copy()
    res_comp['H'] -= 2
    res_comp['O'] -= 1
    return res_comp


class AAComp:
    AAElement = {
        'A': {'C': 3, 'N': 1, 'O': 2, 'H': 7},
    }


class ResComp:
    ResElement = {aa: element_aa2res(aa_comp=aa_comp) for aa, aa_comp in AAComp.AAElement.items()}


class AA:
    AA_3to1 = {
        'Ala': 'A',
        'Cys': 'C',
        'Asp': 'D',
        'Glu': 'E',
        'Phe': 'F',
        'Gly': 'G',
        'His': 'H',
        'Ile': 'I',
        'Lys': 'K',
        'Leu': 'L',
        'Met': 'M',
        'Asn': 'N',
        'Pro': 'P',
        'Gln': 'Q',
        'Arg': 'R',
        'Ser': 'S',
        'Thr': 'T',
        'Val': 'V',
        'Trp': 'W',
        'Tyr': 'Y',
    }
    AA_1to3 = dict(zip(AA_3to1.values(), AA_3to1.keys()))
    AAList_20 = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    # May have same one letter abbr
    ExtendedAA_3to1 = {
        'ALA': 'A',
        'ASX': 'B',
        'CYS': 'C',
        'ASP': 'D',
        'GLU': 'E',
        'PHE': 'F',
        'GLY': 'G',
        'HIS': 'H',
        'ILE': 'I',
        'LYS': 'K',
        'LEU': 'L',
        'Xle': 'L',
        'MET': 'M',
        'ASN': 'N',
        'PYL': 'O',
        'PRO': 'P',
        'GLN': 'Q',
        'ARG': 'R',
        'SER': 'S',
        'THR': 'T',
        'SEC': 'U',
        'VAL': 'V',
        'TRP': 'W',
        'XAA': 'X',
        'TYR': 'Y',
        'GLX': 'Z',
        'OTHER': 'X',
        'TERM': '*'
    }

    Hydro = ['A', 'F', 'I', 'L', 'M', 'P', 'V', 'W']

    # Anal. Chem. 2008, 80, 18, 7036-7042 https://doi.org/10.1021/ac800984n
    Rc_RPLC = '''residue	TFA FA	pH 10
W	13.12	13.67	12.27
F	11.34	11.92	10.19
L	9.44	9.89	8.74
I	7.86	9.06	7.47
M	6.57	6.96	5.67
V	4.86	5.72	4.86
Y	5.4 5.97    4.77
C	0.04	0.7	1.06
P	1.62	1.98	1.85
A	1.11	1.63	1.57
E	1.08	1.75	−4.94
T	0.48	1.37	1.06
D	−0.22	0.95	−5.41
Q	−0.53	0.2	0.3
S	−0.33	0.27	0.61
G	−0.35	−0.07	0.17
R	−2.58	−3.55	3.56
N	−1.44	−0.59	0.04
H	−3.04	−5.05	0.66
K	−3.53	−5.08	2.8'''

    # Anal. Chem. 2017, 89, 11795−11802 DOI:10.1021/acs.analchem.7b03436
    Rc_SCX = '''residue	N-terminal	N+1	N+2	internal	C-2	C-1	C-terminal
R	1.271	1.267	1.217	1.085	1.090	1.095	1.069
H	1.192	1.199	1.162	1.038	1.043	0.980	0.921
K	1.096	1.103	1.043	0.972	0.969	0.953	0.974
W	0.075	0.112	0.125	0.105	0.092	0.101	0.016
N	−0.008	0.004	0.027	0.036	0.033	0.037	0.085
Y	−0.037	0.000	0.019	0.028	0.014	0.018	0.097
G	−0.051	−0.027	0.022	0.028	0.019	0.019	0.134
C	−0.016	0.009	0.015	0.024	0.025	0.009	0.054
F	−0.051	−0.010	0.006	0.020	0.007	0.005	0.004
D	−0.150	−0.043	-0.003	0.012	0.009	0.018	0.031
S	−0.053	−0.031	0.000	0.011	0.007	0.000	0.089
E	−0.081	−0.054	−0.025	0.008	-0.003	0.001	0.041
Q	−0.066	−0.036	−0.018	0.002	−0.013	−0.009	0.078
M	−0.076	−0.055	−0.035	−0.007	−0.033	−0.023	−0.056
A	−0.106	−0.063	−0.032	−0.010	−0.024	−0.022	0.042
T	−0.089	−0.069	−0.037	−0.018	−0.024	−0.019	0.033
L	−0.136	−0.088	−0.058	−0.032	−0.053	−0.040	0.009
I	−0.121	−0.085	−0.068	−0.040	−0.054	−0.049	0.003
V	−0.136	−0.090	−0.060	−0.043	−0.055	−0.045	0.034
P	-0.124	−0.068	−0.062	−0.054	−0.057	−0.056	0.049'''

    def get_rc(self, chro_type='RPLC'):
        import io
        import pandas as pd
        if chro_type == 'RPLC':
            data = self.Rc_RPLC
        elif chro_type == 'SCX':
            data = self.Rc_SCX
        else:
            raise
        return pd.read_csv(io.StringIO(data), sep='\t')


class AAInfo:
    """
    The most useful class for my error-like memory
    """
    AAName = '''Full name	Chinese	3-letter	1-letter
Alanine	丙氨酸	Ala	A
Cysteine	半胱氨酸	Cys	C
Asparticacid	天冬氨酸	Asp	D
Glutamicacid	谷氨酸	Glu	E
Phenylalanine	苯丙氨酸	Phe	F
Glycine	甘氨酸	Gly	G
Histidine	组氨酸	His	H
Isoleucine	异亮氨酸	Ile	I
Lysine	赖氨酸	Lys	K
Leucine	亮氨酸	Leu	L
Methionine	甲硫氨酸	Met	M
Asparagine	天冬酰胺	Asn	N
Proline	脯氨酸	Pro	P
Glutamine	谷氨酰胺	Gln	Q
Arginine	精氨酸	Arg	R
Serine	丝氨酸	Ser	S
Threonine	苏氨酸	Thr	T
Valine	缬氨酸	Val	V
Tryptophan	色氨酸	Trp	W
Tyrosine	酪氨酸	Tyr	Y'''

    def get_aa_name(self):
        import io
        import pandas as pd
        return pd.read_csv(io.StringIO(self.AAName), sep='\t')
