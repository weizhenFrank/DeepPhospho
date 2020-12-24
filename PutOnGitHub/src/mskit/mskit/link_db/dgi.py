import requests
import pandas as pd
import re
import numpy as np


#############################
def todo():
    """
    TODO: 从jupyter整理
    """
#############################


DGIInteractionURL = r'http://dgidb.org/api/v2/interactions.json?genes={}'
KEGGDrugDatabaseFind = r'http://rest.kegg.jp/find/drug/{}'  # Fill in drug name which is connected with plus sign
KEGGGetURL = r'http://rest.kegg.jp/get/{}'  # Fill in drug D number

genename_string = ','.join(['gene list'])
genename_dgi_request = requests.get(DGIInteractionURL.format(genename_string))
interaction_json_parse = genename_dgi_request.json()


print(interaction_json_parse['ambiguousTerms'])
print(interaction_json_parse['unmatchedTerms'])
print(len(interaction_json_parse['unmatchedTerms'][0].split(', ')))



matched_protein_df = protein_list_df[~protein_list_df['GeneName'].str.lower().isin(
    [_.lower() for _ in interaction_json_parse['unmatchedTerms'][0].split(', ')])]

all_info_list = []

for each_dgi_info in interaction_json_parse['matchedTerms']:
    # ---- Here is the basic info from DGI interaction database ----
    search_term = each_dgi_info['searchTerm']
    matched_genename = each_dgi_info['geneName']
    matched_gene_longname = each_dgi_info['geneLongName']
    matched_gene_entrezid = each_dgi_info['entrezId']
    # ---- Here get the info from original excel sheet for further process ----
    each_matched_df = matched_protein_df[matched_protein_df['GeneName'].str.lower() == search_term.lower()]
    protein_accession = each_matched_df.iloc[0]['ProteinAccession']
    genename = each_matched_df.iloc[0]['GeneName']
    protein_description = each_matched_df.iloc[0]['Description']
    membrane_type = each_matched_df.iloc[0]['MPType']
    # ---- Here process all interactions in the requested DGI data for each gene ----
    interaction_list = each_dgi_info['interactions']
    for each_interaction_info in interaction_list:
        drug_interactionid = each_interaction_info['interactionId']
        drug_interaction_types = each_interaction_info['interactionTypes']
        drug_name = each_interaction_info['drugName']
        drug_chemblid = each_interaction_info['drugChemblId']
        drug_sources = each_interaction_info['sources']
        drug_pmids = [str(_) for _ in each_interaction_info['pmids']]
        drug_score = each_interaction_info['score']

        each_complete_data_line = [protein_accession, genename, protein_description, membrane_type,
                                  search_term, matched_genename, matched_gene_longname, matched_gene_entrezid,
                                  drug_interactionid, ';'.join(drug_interaction_types), drug_name, drug_chemblid,
                                  ';'.join(drug_sources), ';'.join(drug_pmids), drug_score]
        all_info_list.append(each_complete_data_line)


dgi_info_df = pd.DataFrame(all_info_list, columns=['ProteinAccession', 'GeneName', 'Description', 'MPType',
                                                  'SearchTerm', 'MatchedGeneName', 'MatchedGeneLongname', 'MatchedGeneEntrezId',
                                                  'InteractionId', 'InteractionType', 'DrugName', 'DrugChemblId',
                                                  'DrugSources', 'DrugPMIDs', 'DrugScore'])
