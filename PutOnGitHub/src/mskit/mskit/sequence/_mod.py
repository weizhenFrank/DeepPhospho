from mskit.constants import BasicModification
from mskit import rapid_kit


class ModOperation(BasicModification):
    def __init__(self):
        """
        self._mod_rule stores the mod rule for peptide. Default is pre-defined StandardModRule.
        self._mod_rule_dict stores manurally defined extra mod rules
        self._display_mod the mods set for assembling queried mod
        self._mod_aa the possible AA for the target mod
        """
        self._mod_rule = self.StandardModRule
        self._mod_rule_dict = self.ModRuleDict
        self._extra_mod_rule_name_list = []

        self._alias_dict = self.ModAliasDict

        self._display_mod = self.StandardMods

        self._mod_aa = self.ModAA

    def __call__(self, mod=None, aa=None):
        """
        To get the mod directly
        """
        return self.query_mod(mod=mod, aa=aa)

    def query_mod(self, mod=None, aa=None):
        """
        Get the mod for display
        Example:
            If mod rule is default standard rule: mod='Carbamidomethyl', aa='C' -> '[Carbamidomethyl (C)]'
            If mod rule is '({mod})' which is used in MaxQuant. mod='ph' -> '(ph)'
        """
        standard_mod = self.get_standard_mod(mod)
        used_mod = self._display_mod[standard_mod]
        return self._mod_rule.format(mod=used_mod, aa=aa)

    def get_standard_mod(self, modname):
        return self._alias_dict[modname]

    def set_extra_mod_rule(self, extra_mod_rule, rule_name='extra_1', set_used=True):
        """
        :param extra_mod_rule
        This receive a string to define the format of mod rule.
        There are two reserved parameters can be used: mod, aa
        Example: '[{mod} (aa)}]', this will define a mod rule like '[Carbamidomethyl (C)]'
        :param set_used
        True means set the currently used mod rule to this extra mod rule
        :param rule_name
        For storing the name of extra mod rule.
        """
        self._mod_rule_dict[rule_name] = extra_mod_rule
        self._extra_mod_rule_name_list.append(rule_name)
        if set_used:
            self._mod_rule = extra_mod_rule

    def add_mod_alias(self, alias_dict: dict, method='direct'):
        """
        :param alias_dict: A dict to define name aliases of modifications. Details are listed in method param.
        :param method: 'direct' or 'list'.
            'direct' to extend the alias dict directly with key-value pairs like: {'alias_mod': 'standard_mod'}
            'list' to extend the alias dict via multi aliases with format: {'standard_mod': ['alias_1', 'alias_2']}
        """
        if method == 'direct':
            for alias, standard in alias_dict.items():
                self._alias_dict[alias] = self._alias_dict[standard]
        elif method == 'list':
            for standard, aliases in alias_dict.items():
                for alias in aliases:
                    self._alias_dict[alias] = self._alias_dict[standard]
        else:
            raise

    def set_display_mod(self, mod_dict: dict, method='add'):
        """
        Define the mods for display (i.e. the mod string for assembly)
        :param mod_dict
        The key of mod_dict is any mod name that can be found in the alias dict,
        and the key will be converted to standard mod name.
        Then the value will be a new display mod name.
        e.g. mod_dict={'ox': 'oxi'} -> from alias, 'ox' is 'Oxidation', then the mod for mod rule will be 'oxi'
        :param method
        'add' means add the new display mods and the shared mods will be replaced to new ones
        'new' means the new display mod dict will completely overwrite the previous one
        """
        _mod_dict = dict()
        for mod, disp_mod in mod_dict.items():
            standard_mod = self.get_standard_mod(mod)
            _mod_dict[standard_mod] = disp_mod
        if method == 'add':
            self._display_mod = {**self._display_mod, **_mod_dict}
        elif method == 'new':
            self._display_mod = _mod_dict
        else:
            raise

    def extend_mod(self, mod):
        """
        This receive mod to extend the pre-defined mods
        Example:
            mod='NewMod' -> self._display_mod['NewMod'] = 'NewMod' and self._alias_dict['NewMod'] = 'NewMod'
            This will add the mod itself for new query
        :param mod New mod name
        """
        self._display_mod[mod] = mod
        self._alias_dict[mod] = mod

    def switch_mod_rule(self, rule=None):
        if rule == 'standard':
            self._mod_rule = self._mod_rule_dict['standard']
        else:
            if rule:
                try:
                    self._mod_rule = self._mod_rule_dict[rule]
                except KeyError:
                    print('No extra mod rule defined, change to standard mod rule')
                    self._mod_rule = self._mod_rule_dict['standard']
            else:
                rule = self._extra_mod_rule_name_list[-1]
                self._mod_rule = self._mod_rule_dict[rule]

    def show_extra_mod_rule_list(self):
        return self._extra_mod_rule_name_list

    def return_mod_rule(self):
        return self._mod_rule

    def add_mod(self, pep, mod):
        mod_pep = rapid_kit.add_mod(pep=pep, mod=mod, mod_processor=self)
        return mod_pep

    def query_possible_aa(self, mod_name):
        standard_mod = self.get_standard_mod(mod_name)
        return self._mod_aa[standard_mod]

    def __extract_mod(self, mod_seq):
        pass

    def __trans_mod(self, modpep):
        pass
