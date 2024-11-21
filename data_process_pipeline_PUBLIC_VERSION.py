"""
***POSTED FOR PUBLIC DEMONSTRATION OF MY PROJECT DURING JOB SEARCH***


Pipeline to pull, parse, and prepare a dataset for training a logistic regression classifier to detect if postretraction citations--citations that
are made to retracted articles after the article retraction notice has been posted--are acknowledging the retraction or not. 

As of 11/21/24, still in early development stages. Full list of dependencies, unit tests, and troubleshooting to be updated soon

Written by Christian Sodano. Contact cnsodano@gmail.com for information.

Code heavily relies on Titipat Achakulvisut's excellent package pubmed_parser which can be found here: https://github.com/titipata/pubmed_parser. 
Conceptual inspiration for project heavily relies on Hsiao & Schneider 2022 (https://pmc.ncbi.nlm.nih.gov/articles/PMC9520488/)

Draws data from pubmed's open access collection using their NCBI eutils, and the data in Hsiao & Schneider 2022 (https://databank.illinois.edu/datasets/IDB-3213475)

"""

import random  # To solve NaN problem, see below
import pandas as pd
import subprocess  # To programmatically pull nxml files for each article using an external script
from find_citing_par_LI import find_citing_par  # Will be used to match a sentence with a paragraph context, once the file is downloaded


DATA_ROOT = "csvs/" 
UNKNOWING_POSTRETRACTION_CITATIONS_FILE_NAME = "PubMed_retracted_publication_CitCntxt_withYR_v3.csv"
KNOWING_POSTRETRACTION_CITATIONS_FILE_NAME = "724_knowingly_post_retraction_cit.csv"
NUM_DATASET_SIZE = 724  # Because we have 724 'known' postretraction citations (Semimanually annotated ground truth) (https://databank.illinois.edu/datasets/IDB-3213475)
SEED = random.seed(43) # So I can make changes without pulling different subset of articles 
print(random.random()) # Should be 0.038551839337380045
breakpoint()

# Pulling docs that cite retracted articles and are not annotated by human to be "knowing" (may include some false negatives), includes some docs that are published before the retraction notice
citations_to_retracted_papers  = pd.read_csv(DATA_ROOT + UNKNOWING_POSTRETRACTION_CITATIONS_FILE_NAME)  # Requires "csvs/" folder structure to be where you stored the data from https://databank.illinois.edu/datasets/IDB-3213475
post_retraction_indices = citations_to_retracted_papers['post_retraction'] == 1  # Only pull postretraction citations (identifying knowing vs. unknowing for citations to articles that later got retracted before they were retracted (i.e. *pre*-retraction) would be a trivial and unhelpful task)

# Find those postretraction citing papers
citations_to_retracted_papers = citations_to_retracted_papers[post_retraction_indices]  # Also known as "unknowing_postretraction_citations", as this PubMed_retracted_publication_CitCntxt_withYR_v3.csv file does not include knowing documents

# Pulling data from csv into variables to be saved with my preferred names
indices = random.sample(list(range(len(citations_to_retracted_papers))), NUM_DATASET_SIZE); # Take only 724 of the unknowing post-retraction citations. #### Option to make more sophisticated to improve training. Simply pulling 724 to balance may not be best.
document_PMIDs = citations_to_retracted_papers.iloc[indices]["pmid"]
document_PMCIDs =  citations_to_retracted_papers.iloc[indices]["pmcid"]
document_citation_contexts = citations_to_retracted_papers.iloc[indices]["citation"]
retracted_papers_PMIDs_cited_by_documents =  citations_to_retracted_papers.iloc[indices]["intxt_pmid"]
citation_context_section_location = citations_to_retracted_papers.iloc[indices]["IMRaD"]
publication_year = citations_to_retracted_papers.iloc[indices]["year"]
retracted_year = citations_to_retracted_papers.iloc[indices]["retracted_yr"]


# Init the dict I will use as the skeleton for dataframe structure
dataframe_dict_init_without_authors = {"postretraction_citing_paper_PMID": [],"postretraction_citing_paper_PMCID":[], "retracted_paper_it_is_citing_PMID":[], "citation_context":[]}
# Build dataframe with preferred column names. "Without authors" is a remnant of a previous attempt to pull the authors from the intext citation
data_without_authors = pd.DataFrame.from_dict(dataframe_dict_init_without_authors)
data_without_authors["postretraction_citing_paper_PMID"] = document_PMIDs
data_without_authors["postretraction_citing_paper_PMCID"] = document_PMCIDs
data_without_authors["retracted_paper_it_is_citing_PMID"] = retracted_papers_PMIDs_cited_by_documents
data_without_authors["citation_context"] = document_citation_contexts
data_without_authors["citation_context_section_location"] = citation_context_section_location
data_without_authors["publication_year"] = publication_year
data_without_authors["retracted_year"] = retracted_year

# OPTIONAL Save checkpoint
SAVE_NAME = "unknowing_data_checkpoint_11-21-24.csv" 
SAVE_PATH = DATA_ROOT + SAVE_NAME 
#data_without_authors.to_csv(SAVE_PATH)
breakpoint()

# Return from checkpoint
unknown_postretraction_citations = pd.read_csv(SAVE_PATH, dtype={"postretraction_citing_paper_PMID":str,"postretraction_citing_paper_PMCID":str,"retracted_paper_it_is_citing_PMID":str,"citation_context":str,"citation_context_section_location":str, "publication_year":int,"retracted_year":int})
unknown_citations_ground_truth_sentences: list[str] = unknown_postretraction_citations["citation_context"].tolist()  # Use to find the paragraph the sentence appears in later
unknown_cit_pmcids = unknown_postretraction_citations["postretraction_citing_paper_PMCID"].tolist()  # As above


# Pulling provided "knowing" postretraction dataset
known_postretraction_citations = pd.read_csv(DATA_ROOT+KNOWING_POSTRETRACTION_CITATIONS_FILE_NAME, dtype={"pmcid":str, "pmid":str, "intxt_pmid": str, "IMRaD": str, "year":int, "retracted_yr":int})
known_citations_ground_truth_sentences: list[str] = known_postretraction_citations["citation"].tolist()
known_cit_pmcids = known_postretraction_citations['pmcid'].tolist()


## Init dicts to turn into dataframes that will be saved as csv 
unknown_features_dict = {"pmcid":unknown_postretraction_citations["postretraction_citing_paper_PMCID"].tolist(),
                      "year_published": unknown_postretraction_citations["publication_year"].tolist(),
                      "year_retracted_article_citing_was_retracted": unknown_postretraction_citations["retracted_year"].tolist(),  
                      "years_between_retraction_and_citation": ( unknown_postretraction_citations["publication_year"]-unknown_postretraction_citations["retracted_year"]).tolist(),
                      "citation_context_IMRaD_section": unknown_postretraction_citations["citation_context_section_location"],
                      "citation_from_csv": unknown_citations_ground_truth_sentences,
                      "citation_paragraph_parsed": []  # Placeholder, will be updated during while loop below
                      }

known_features_dict = {"pmcid":known_postretraction_citations["pmcid"].tolist(),
                      "year_published": known_postretraction_citations["year"].tolist(),
                      "year_retracted_article_citing_was_retracted":known_postretraction_citations["retracted_yr"].tolist(), 
                      "years_between_retraction_and_citation": ( known_postretraction_citations["year"]-known_postretraction_citations["retracted_yr"]).tolist(),
                      "citation_context_IMRaD_section": known_postretraction_citations["IMRaD"],
                      "citation_from_csv": known_citations_ground_truth_sentences,
                      "citation_paragraph_parsed": known_postretraction_citations["longer_context"].tolist() # THIS IS A SHORTCUT FORN OW, ULTIMATELY WANT TO DO THE SAME PARSING OF THE WHOLE 
                                                                                                             # PARAGRPH FOR THE KNOWN AS WELL AS THE UNKNOWN. The known has a "longer context" 
                                                                                                             # that is more than 1 sentence, but not quite the full paragraph
                      }

# After running this once, don't need to do so anymore. This will take a while, downloading 724 files from the internet. 
HELPER_SCRIPT_PATH = "citing_docs_xmls/"
FETCH_NXML_SCRIPT_PATH = "fetch_nxml3.py"  
# NOTE the below will download an nxml file for each PMCID you pass it, and specifically it's set to put it in a nxmls/ directory. TODO update later to pass the directory as an arg
#subprocess.run(["python",HELPER_SCRIPT_PATH + FETCH_NXML_SCRIPT_PATH "--pmcids"] + unknown_cit_pmcids)  # RETRIEVE nxml FILES FOR EACH PMCID, 724 of them
breakpoint()

# Loop through pmcids of unknowing docs and find full paragraph they appeared in.
i: int = 0
i = 0
pbar = tqdm.tqdm(total=len(unknown_cit_pmcids))  # TODO Fix progressbar
while (i<len(unknown_cit_pmcids)):
    pmcid_path = "nxmls/" + unknown_features_dict["pmcid"][i]+".nxml"
    ground_truth_sentence = unknown_citations_ground_truth_sentences[i]
    paragraph_context_of_GT_sentence = find_citing_par(pmcid_path, ground_truth_sentence)  # TODO return match confidence to use as a feature in the logistic regression
    unknown_features_dict["citation_paragraph_parsed"].append(paragraph_context_of_GT_sentence)
    i+=1
pbar.close()
breakpoint()



# TODO:
# If there are any NaNs, remove from unknown features, pull new unknowing postretraction docs, find the par context, and check again for unknowing.
# Ensure not the same indices as before (i.e. check with indices variable to make sure no duplicates)

PARSED_DATA_ROOT = "parsed_data-11-21-24/"
UNKNOWN_FEATURES_NAME = "unknown_features_11-21-24.csv"
unknown_features_df = pd.DataFrame.from_dict(unknown_features_dict) 
unknown_features_df.to_csv(PARSED_DATA_ROOT+UNKNOWN_FEATURES_NAME)

KNOWN_FEATURES_NAME = "known_features_11-21-24.csv"
known_features_df = pd.DataFrame.from_dict(known_features_dict)
known_features_df.to_csv(PARSED_DATA_ROOT+KNOWN_FEATURES_NAME)
breakpoint()

# TODO drop the NaN citation paragraph contexts 
# Something like: 
unknown_features_df.drop(unknown_features_df[unknown_features_df["citation_paragraph_parsed"].isna()].index)
UNKNOWN_FEATURES_NA_DROPPED_NAME = "unknown_features_11-21-24_NA_dropped.csv"
unknown_features_df.to_csv(PARSED_DATA_ROOT+UNKNOWN_FEATURES_NA_DROPPED_NAME)
breakpoint()

## Prepare data for logistic regression
unknowing = pd.read_csv(PARSED_DATA_ROOT+UNKNOWN_FEATURES_NA_DROPPED_NAME)
unknowing_label = pd.Series("unknowing", index=range(len(unknowing)))  # Label to use as target for log regression
unknowing.insert(0,"label",unknowing_label)
#breakpoint()

knowing = pd.read_csv(PARSED_DATA_ROOT+KNOWN_FEATURES_NAME)
drop_indices = random.sample(sorted(knowing.index),724-len(unknowing)) # So training dataset has balanced label distribution
knowing = knowing.drop(drop_indices)
knowing_label = pd.Series("knowing", index=knowing.index)
knowing.insert(0,"label",knowing_label)
#breakpoint()

full_data = pd.concat([knowing,unknowing])
full_data = full_data.sample(frac=1) # Randomizing order so that I can use k-fold cross validation without fear of highly unrepresentative folds
#full_data.to_csv("full_data_cross_val.csv")
FULL_DATA_NAME = "full_data_cross_val_text_only_NAN_dropped_11-21-24.csv"
full_data.to_csv(PARSED_DATA_ROOT+FULL_DATA_NAME)
breakpoint()

# Return, want to normalize numerical data features and add nominal year features as separate features, so that I can capture any year-level abnormalities
# (e.g. articles in years around 2010 may be more likely to be "knowing" due to that year being when the highly publicized Wakefield vaccine article was retracted)
base_data = pd.read_csv(PARSED_DATA_ROOT+FULL_DATA_NAME)
base_data = full_data.copy() 

# nominal and max/min for both year pub and year retract
year_publ_nominal = base_data["year_published"].copy() 
year_publ_min_max_normalized = base_data["year_published"].copy()
year_publ_min_max_normalized = (year_publ_min_max_normalized - min(year_publ_min_max_normalized))/(max(year_publ_min_max_normalized)-min(year_publ_min_max_normalized)); 
year_retracted_nominal = base_data["year_retracted_article_citing_was_retracted"].copy() 
year_retracted_min_max_normalized = base_data["year_retracted_article_citing_was_retracted"].copy()
year_retracted_min_max_normalized = (year_retracted_min_max_normalized - min(year_retracted_min_max_normalized))/(max(year_retracted_min_max_normalized) - min(year_retracted_min_max_normalized))

# Capture some concept of "distance between retraction and the publication of the postretraction citing doc" to leverage the likely relationship; whether positive or negative I'm not sure at the outset
years_btwn_cit_and_retraction_min_max_normalized = base_data["years_between_retraction_and_citation"]
years_btwn_cit_and_retraction_min_max_normalized = (years_btwn_cit_and_retraction_min_max_normalized - min(years_btwn_cit_and_retraction_min_max_normalized))/(max(years_btwn_cit_and_retraction_min_max_normalized) - min(years_btwn_cit_and_retraction_min_max_normalized))
breakpoint()

new_data = base_data[["citation_context_IMRaD_section","citation_paragraph_parsed"]].copy()  # Original datasets offer IMRaD section, including as a feature)
new_data.insert(0,"years_btwn_cit_and_retraction_min_max_normalized", years_btwn_cit_and_retraction_min_max_normalized)
new_data.insert(0,"year_retracted_min_max_normalized", year_retracted_min_max_normalized)
new_data.insert(0,"year_retracted_nominal", year_retracted_nominal)
new_data.insert(0,"year_publ_min_max_normalized", year_publ_min_max_normalized)
new_data.insert(0,"year_publ_nominal", year_publ_nominal)
breakpoint()
new_data.insert(0,"label", base_data["label"])
breakpoint()
PROCESSED_DATA_NAME = "aggregated_data-11-21-24_min_max_normalized_only_features.csv"
new_data.to_csv(DATA_ROOT+PROCESSED_DATA_NAME)