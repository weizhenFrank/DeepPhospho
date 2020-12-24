from .sn_constant import *

import re

from mskit import rapid_kit


class ModProcess(object):
    @staticmethod
    def add_mod(pep: str, mod_type=None):
        """
        Add fixed modification 'C[Carbamidomethyl (C)]'
        Variable modificaion: only M[Oxidation (M)] is supported now, only one oxidation mod will be added when MOD is ['M']
        TODO: Add other modifications and change modification number
        :param pep:
        :param mod_type:
        :return:
        """
        modpep_list = []
        mod_pep = pep.replace('C', MOD.ModDict['C'])

        if mod_type == ['M']:
            if 'M' in mod_pep:
                for _ in re.finditer('M', mod_pep):
                    m_site = _.start()
                    modpep_list.append(mod_pep[:m_site] + MOD.ModDict['M'] + mod_pep[m_site + 1:])
            else:
                modpep_list.append(mod_pep)
        else:
            modpep_list.append(mod_pep)

        return modpep_list

    @staticmethod
    def modpep2intmod():
        """
        C[Carbamidomethyl (C)] -> C[+16]
        int mod pep is not the necessary column in spectronaut library, so it is not used now
        :return:
        """
        pass

    @staticmethod
    def remove_str_mod(mod_pep):
        pep = mod_pep.replace(MOD.ModDict['C'], 'C').replace(MOD.ModDict['M'], 'M')
        return pep

    @staticmethod
    def remove_int_mod(mod_pep):
        pep = mod_pep.replace('1', 'M')
        return pep

    @staticmethod
    def mod2new(old_mod_pep):
        new_mod_pep = old_mod_pep.replace(MOD.ModDict_old['C'], MOD.ModDict_new['C']).replace(MOD.ModDict_old['M'], MOD.ModDict_new['M'])
        return new_mod_pep

    @staticmethod
    def mod2old(new_mod_pep):
        old_mod_pep = new_mod_pep.replace(MOD.ModDict_new['C'], MOD.ModDict_old['C']).replace(MOD.ModDict_new['M'], MOD.ModDict_old['M'])
        return old_mod_pep

    @staticmethod
    def mod_num2str(num_mod_pep):
        str_mod_pep = num_mod_pep.replace('C', MOD.ModDict['C']).replace('1', MOD.ModDict['M'])
        return str_mod_pep

    @staticmethod
    def mod_str2num(str_mod_pep):
        num_mod_pep = str_mod_pep.replace(MOD.ModDict['C'], 'C').replace(MOD.ModDict['M'], '1')
        return num_mod_pep

