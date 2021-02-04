from .data_struc_kit import str_mod_to_list

import re


def get_title_dict(title_content: str) -> dict:
    title_dict = dict([(__, _) for _, __ in enumerate(
        title_content.strip('\n').split('\t'))])
    return title_dict


def semicolon_combination(*sn, keep_order=False):
    """
    This will combine two strings with semicolons and drop duplication
    Example: s1='Q1;Q2', s2='Q2;Q3' -> 'Q1;Q2;Q3'
    Note that the order may change if keep_order=False
    """
    s_list = map(lambda _: _.strip(';').split(';'), sn)
    flatten_s = list(filter(lambda x: True if x else False, sum(s_list, [])))
    unique_s = list(set(flatten_s))
    if keep_order:
        unique_s = sorted(unique_s, key=flatten_s.index)
    return ';'.join(unique_s)


def extract_bracket(str_with_bracket, ):  # Need a parameter to choose to use () or [] or others (by manurally define?) and a parameter to skip how many additional brackets
    bracket_start = [left_bracket.start() for left_bracket in re.finditer('\(', str_with_bracket)]  # Add [::2] if there is one additional bracket in the expected one
    bracket_end = [right_bracket.start() for right_bracket in re.finditer('\)', str_with_bracket)]  # Add [1::2] if add the additional operation at the last step
    return bracket_start, bracket_end


def split_fragment_name(fragment_name):
    frag_type, frag_num, frag_charge, frag_loss = re.findall(
        '([abcxyz])(\\d+)\\+(\\d+)-?(.*)', fragment_name)[0]
    return frag_type, int(frag_num), int(frag_charge), frag_loss


def split_prec(prec: str, keep_underline=False):
    modpep, charge = prec.split('.')
    if not keep_underline:
        modpep = modpep.replace('_', '')
    return modpep, int(charge)


def assemble_prec(modpep, charge):
    if not modpep.startswith('_'):
        modpep = f'_{modpep}_'
    return f'{modpep}.{charge}'


def split_mod(modpep, mod_ident='bracket'):
    if mod_ident == 'bracket':
        mod_ident = ('[', ']')
    elif mod_ident == 'parenthesis':
        mod_ident = ('(', ')')
    else:
        pass
    re_find_pattern = '(\\{}.+?\\{})'.format(*mod_ident)
    re_sub_pattern = '\\{}.*?\\{}'.format(*mod_ident)
    modpep = modpep.replace('_', '')
    mod_len = 0
    mod = ''
    for _ in re.finditer(re_find_pattern, modpep):
        _start, _end = _.span()
        mod += '{},{};'.format(_start - mod_len, _.group().strip(''.join(mod_ident)))
        mod_len += _end - _start
    stripped_pep = re.sub(re_sub_pattern, '', modpep)
    return stripped_pep, mod


def add_mod(pep, mod, mod_processor):
    """
    mod_process is the ModOperation class
    """
    if mod:
        if isinstance(mod, str):
            mod = str_mod_to_list(mod)
        mod = sorted(mod, key=lambda x: x[0])
        mod_pep_list = []
        prev_site_num = 0
        for mod_site, mod_name in mod:
            mod_pep_list.append(pep[prev_site_num: mod_site])
            if mod_site != 0:
                mod_aa = mod_pep_list[-1][-1]
            else:
                mod_aa = pep[0]
            mod_pep_list.append(mod_processor(mod=mod_name, aa=mod_aa))
            prev_site_num = mod_site
        mod_pep_list.append(pep[prev_site_num:])
        mod_pep = ''.join(mod_pep_list)
    else:
        mod_pep = pep
    mod_pep = f'_{mod_pep}_'
    return mod_pep


def fasta_title(title: str, title_type='uniprot'):
    title = title.lstrip('>')

    if '|' in title:
        ident = title.split('|')[1]
    else:
        ident = title.split(' ')[0]
    return ident


def join_seqtext_in_fasta(seq_text):
    return ''.join(seq_text.split('\n'))


def join_seqlines_in_fasta(seq_lines):
    return ''.join([_.strip('\n') for _ in seq_lines])
