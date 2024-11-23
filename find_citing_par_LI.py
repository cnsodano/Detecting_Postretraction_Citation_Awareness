"""
A script to return the paragraph context of a sentence that appears in an article. Used to match the paragraph context of a citation sentence that includes
a citation to a retracted article, that the full paragraph context might be used to determine if the citation is acknowledging the article's retracted
status or not. 

Script is nearly entirely adopted from Titipat Achakulvisut's pubmed_parser package, with slight adjustments to add fuzzy matching. All credit to them.
link: https://github.com/titipata/pubmed_parser/blob/master/pubmed_parser/pubmed_oa_parser.py

As of 11/23/2024, project is still in development. 

This script is called by data_process_pipeline script 
"""

import os
from lxml import etree
from itertools import chain
from pubmed_parser.utils import read_xml, stringify_affiliation_rec, stringify_children
from unidecode import unidecode
from thefuzz import fuzz as fuzzy  # For fuzzy comparison of strings
from thefuzz import process as fuzzyprocess


def find_citing_par(path, sentence, confidence_threshold):
    """
    Give path to a given PubMed OA file and a sentence string, parse and return
    the single paragraph it belongs to, or None if there isn't a clear match due to 

    Parameters
    ----------
    path: str
        A string to an XML path.

    sentence: str
        A string of the sentence of natural language text that you want to extract the context for

    confidence_threshold: int
        An int of the fuzzy matching confidence value that you want to use to greedily determine the matching paragraph. 
        A higher value will increase precision, but may reduce recall (you will have some return NaN). If you are using 
        documents where the document is rife with the same terms in the query sentence, you may want to increase the threshold


    Return
    ------
    paragraph_text: str
        A string of the paragraph context in which the sentence passed occurs. This is determined by a greedy approach using fuzzy matching
        and will return the first paragraph that matches at a fuzzy matching confidence of *confidence_threshold* or higher. Not the ideal solution,
        but works for now in development
    """
    tree = read_xml(path)
    paragraphs = tree.xpath("//body//p")
    for paragraph in paragraphs:
        
        paragraph_text = stringify_children(paragraph)

        # Break into sentences
        paragraph_sentences: list[str] = paragraph_text.split('.')[0:-1]  # 0:-1 removes blank string at end of list

        sentence_to_match = sentence

        for query_sentence in paragraph_sentences:
            match = fuzzy.ratio(sentence_to_match,query_sentence) 
            if (match > 85): # Only matches if match confidence is >85. Check for false negatives before using.
                return paragraph_text
    return None
