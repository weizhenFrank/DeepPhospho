"""
ProteomeXchange
    Pride

"""


class ProteomeXchange:
    class Pride:
        """
        ListQueryUrl:
            Url + Project ID (PXD...)
                e.g. https://www.ebi.ac.uk/pride/ws/archive/file/list/project/PXD004732
            This will get a json file
        Example FTP file address
            ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2018/06/PXD009449/MaxQuant_1.5.3.30.zip
        """
        ListQueryUrl = 'https://www.ebi.ac.uk/pride/ws/archive/file/list/project/'
        FTP_Url = 'ftp://ftp.pride.ebi.ac.uk'
        ASP_Url = 'era-fasp@fasp.pride.ebi.ac.uk:'
        FTP_IP = '193.62.197.74'
        PRIDE_StoragePrefix = '/pride/data/archive/'


class DGI:
    """
    Drug Gene Interaction: http://www.dgidb.org/
    """
    DGIInteractionURL = r'http://dgidb.org/api/v2/interactions.json?genes={}'


class KEGG:
    KEGGDrugDatabaseFind = r'http://rest.kegg.jp/find/drug/{}'  # Fill in drug name which is connected with plus sign
    KEGGGetURL = r'http://rest.kegg.jp/get/{}'  # Fill in drug D number


class STRING:
    STRING_API_URL = "https://string-db.org/api"
    Methods = ['interaction_partners']
    OutputFormat = []
