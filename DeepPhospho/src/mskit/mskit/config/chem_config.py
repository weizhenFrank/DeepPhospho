import re


class ChemRE:
    FormulaElement = re.compile(r'([A-Z][a-z]?)(\d*)')
