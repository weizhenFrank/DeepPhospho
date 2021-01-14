"""
In class XLer

XLers:
    A dict contains all properties of cross-linkers (XLers)
        Key: The abbr name of XLers with all uppercase letters except some special name, e.g. Sulfo-GMBS

FullToAbbr:
    Search the full name of XLers to their abbr

"""


class XLer:
    XLers = {
        'BS3': {
            'FullName': 'BS3',
            'Site-1': ['K', 'N-term'],
            'Site-2': ['K', 'N-term'],
            'Compound': 'C16H18N2Na2O14S2',
            'Composition': '',
            'MaxDistance': 35.
        },

        'DSS': {
            'FullName': 'DISUCCINIMIDYL SUBERATE',
            'Site-1': ['K', 'N-term'],
            'Site-2': ['K', 'N-term'],
            'Compound': 'C16H20N2O8',
            'Composition': '',
            'MaxDistance': 35.
        },

        'EGS': {
            'FullName': 'EGS',
            'Site-1': ['K', 'N-term'],
            'Site-2': ['K', 'N-term'],
            'Compound': 'C18H20N2O12',
            'Composition': '',
            'MaxDistance': 40.
        },

        'EDC': {
            'FullName': 'EDC',
            'Site-1': ['K', 'N-term'],
            'Site-2': ['D', 'E'],
            'Compound': '',
            'Composition': '',
            'MaxDistance': 25.
        },

        'PDH': {
            'FullName': 'PDH',
            'Site-1': ['D', 'E', 'N-term'],
            'Site-2': ['D', 'E'],
            'Compound': '',
            'Composition': '',
            'MaxDistance': 35.
        },

        'Sulfo-GMBS': {
            'FullName': 'Sulfo-GMBS',
            'Site': {
                'Site-1': ['K'],  # TODO Not sure this site
                'Site-2': ['C'],
            },
            'Compound': 'C12H12N2O9S',
            'Composition': '',
            'MaxDistance': 31.
        },

        'ArGO': {
            'FullName': 'ArGO',
            'Site-1': ['R', 'N-term'],
            'Site-2': ['R'],
            'Compound': '',
            'Composition': '',
            'MaxDistance': 43.
        },

        'KArGO': {
            'FullName': 'KArGO',
            'Site-1': ['K', 'N-term'],
            'Site-2': ['R'],
            'Compound': '',
            'Composition': '',
            'MaxDistance': 41.
        },
    }

    FullToAbbr = {desc['FullName']: abbr for abbr, desc in XLers.items()}

    LinkType = {
        'Amine-Amine': ['DSS', 'BS3', 'EGS'],
        'Amine-Carboxyl': ['EDC'],
        'Carboxyl-Carboxyl': ['PDH'],
        'Amine-Sulfhydryl': ['Sulfo-GMBS'],

    }
