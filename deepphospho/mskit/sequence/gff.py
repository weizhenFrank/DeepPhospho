import os


class GFF3(object):
    """

    https://github.com/The-Sequence-Ontology/Specifications/blob/master/gff3.md

    Generic Feature Format Version 3

    Column 1: "seqid"
    Column 2: "source"
    Column 3: "type"
    Columns 4 & 5: "start" and "end"
    Column 6: "score"
    Column 7: "strand"
    Column 8: "phase"
    Column 9: "attributes"

    Predefined tags in col 9 (attributes)
        ID
        Name
        Alias
        Parent
        Target
        Gap
        Derives_from
        Note
        Dbxref
        Ontology_term
        Is_circular

    Characters with reserved meanings in column 9
        ; semicolon (%3B)
        = equals (%3D)
        & ampersand (%26)
        , comma (%2C)

    Attributes which can have multiple values (split by comma - ,)
        Parent, Alias, Note, Dbxref ,and Ontology_term

    Attribute names are case sensitive.
        Parent != parent

    Lines beginning with '##' are directives (sometimes called pragmas or meta-data) and provide meta-information about the document as a whole.
    Blank lines should be ignored by parsers and lines beginning with a single '#' are used for human-readable comments and can be ignored by parsers.
    End-of-line comments (comments preceded by # at the end of and on the same line as a feature or directive line) are not allowed.

    The Gap Attribute - Eg. Gap=M8 D3 M6 I1 M6
        Code	Operation
        M	match
        I	insert a gap into the reference sequence
        D	insert a gap into the target (delete from reference)
        F	frameshift forward in the reference sequence
        R	frameshift reverse in the reference sequence

    ###
        This directive (three # signs in a row) indicates that all forward references to feature IDs that have been seen to this point have been resolved.
        After seeing this directive, a program that is processing the file serially can close off any open objects that it has created and return them,
        thereby allowing iterative access to the file.

    ##FASTA
        This notation indicates that the annotation portion of the file is at an end and that the remainder of the file contains one or more sequences
        (nucleotide or protein) in FASTA format.
        This allows features and sequences to be bundled together.
        All FASTA sequences included in the file must be included together at the end of the file and may not be interspersed with the features lines.
        Once a ##FASTA section is encountered no other content beyond valid FASTA sequence is allowed.
    """

    def __init__(self, gff_path=None, parse_with_init=True):
        self.file_path = gff_path

        if self.file_path:
            self._check_path()

        self.comment_lines = []

        self._have_fasta = False
        self.fasta_lines = None

        if parse_with_init:
            self.parse()

    def _check_path(self):
        if not os.path.exists(self.file_path):
            print(f'The path of input GFF file is not valid: {self.file_path}')
            raise FileNotFoundError

    def set_file_path(self, gff_path):
        self.file_path = gff_path
        self._check_path()

    def parse(self):
        with open(self.file_path, 'r') as f:
            for row in f:
                row = row.strip('\n')
                if row.startswith('#'):
                    self.comment_lines.append(row)

                    if row == '##FASTA':
                        self._have_fasta = True
                        self.fasta_lines = []

                elif row == '###':
                    pass
                else:
                    if self._have_fasta:
                        self.fasta_lines.append(row)
                    else:
                        pass

    def a(self, gff_path):
        gff_dict = dict()
        with open(gff_path, 'r') as f:
            _ = f.readline()
            trans_sites = []
            for r in f:
                if r.startswith('##'):
                    if trans_sites:
                        trans_sites.append(end_site)
                        gff_dict[acc] = trans_sites
                    trans_sites = []
                    split_row = r.split(' ')
                    acc = split_row[1]
                    end_site = int(split_row[3])
                else:
                    split_row = r.split('\t')
                    if split_row[2] == 'Transmembrane':
                        trans_sites.append((int(split_row[3]), int(split_row[4])))
                    else:
                        continue
            trans_sites.append(end_site)
            gff_dict[acc] = trans_sites

    def with_fasta(self):
        if self._have_fasta:
            return True
        else:
            return False
