"""

"""

"""
### Description for supported modifications ###

DeepPhospho supports phosphorylation and common modifications listed as below:
    Phospho on S/T/Y
    Oxidation on M
    Acetyl at peptide N-term
    And all C in peptides are regarded as C with Carbamidomethyl

### Description of different modified peptide formats ###

The supported modified peptide formats are listed as below:
    DeepPhospho (modified residues with integers annotated used as DeepPhospho input):
        2/3/4 for S/T/Y with Phospho (STY), respectively
        1 for Oxidation (M)
        * as peptide first char for Acetyl (N-term)
        @ as peptide first char for Non-Acetyl modified
        Example: *1ED2MCLK == (ac)M(ox)EDS(ph)MCLK
        Example: @Q3D2MCLK == QT(ph)DS(ph)MCLK

    Spectronaut 13+ style (modifications with long-term name start and end with brackets):
        [Phospho (STY)] for Phospho (STY)
        [Oxidation (M)] for Oxidation (M)
        [Acetyl (Protein N-term)] for Acetyl (N-term)
        Example: _[Acetyl (Protein N-term)]M[Oxidation (M)]LSLRVPLAPITDPQQLQLS[Phospho (STY)]PLK_

    MQ 1.5- style (modifications with lowercase shorthand):
        (ph) for Phospho (STY)
        (ox) for Oxidation (M)
        (ac) for Acetyl (N-term)
        Example: _(ac)GS(ph)QDM(ox)GS(ph)PLRET(ph)RK_

    MQ 1.6+ style (modifications with long-term name start and end with parentheses):
        (Phospho (STY)) for Phospho (STY)
        (Oxidation (M)) for Oxidation (M)
        (Acetyl (Protein N-term)) for Acetyl (N-term)
        Example: _(Acetyl (Protein N-term))TM(Oxidation (M))DKS(Phospho (STY))ELVQK_

    Comet style (modifications with symbols annotated):
        S@/T@/Y@ for Phospho (STY)
        M* for Oxidation (M)
        n# for Acetyl (N-term)
        Example: n#DFM*SPKFS@LT@DVEY@PAWCQDDEVPITM*QEIR
        
    The formats of modified peptide format of SN 12- is a little different with 13+, and we didn't support it because the support of PTM data is much better from 13+
    If you need other format support, please contact us and we will add the support of it
"""

from deep_phospho.proteomics_utils.post_analysis import maxquant, spectronaut, comet


def sn13_to_intpep(x):
    return spectronaut.sn_modpep_to_intseq(x)


def mq1_5_to_intpep(x):
    return maxquant.mq_modpep_to_intseq_1_5(x)


def mq1_6_to_intpep(x):
    return maxquant.mq_modpep_to_intseq_1_6(x)


def comet_to_intpep(x):
    return comet.comet_to_intseq(x)


def deepphospho_to_intpep(x):
    return x


def intpep_to_sn13(x):
    return spectronaut.intseq_to_sn_modpep(x)

