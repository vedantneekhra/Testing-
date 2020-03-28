"""
Text Preprocessing
  Noise Removal
  Lexicon Normalization
  Lemmatization
  Stemming
  Object Standardization

Text to Features (Feature Engineering on text data)
  Syntactical Parsing
    Dependency Grammar
    Part of Speech Tagging
  Entity Parsing
    Phrase Detection
    Named Entity Recognition
    Topic Modelling
    N-Grams
  Statistical features
    TF – IDF
    Frequency / Density Features
    Readability Features
    Word Embeddings

Important tasks of NLP
  Text Classification
  Text Matching
    Levenshtein Distance
    Phonetic Matching
    Flexible String Matching
  Coreference Resolution
  Other Problems
"""






#beautifulsoup
#nltk
#spacy
#unicodedata
#re
import spacy
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re
from bs4 import BeautifulSoup
from contractions import CONTRACTION_MAP
import unicodedata
nlp = spacy.load('en_core', parse=True, tag=True, entity=True)
#nlp_vec = spacy.load('en_vecs', parse = True, tag=True, #entity=True)
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text

strip_html_tags('<html><h2>Some important text</h2></html>')

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

remove_accented_chars('Sómě Áccěntěd těxt')

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

expand_contractions("Y'all can't expand contractions I'd think")

def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

remove_special_characters("Well this was fun! What do you think? 123#@!", 
                          remove_digits=True)

# Stemming
# Word stems are also known as the base form of a word, and we can create 
# new words by attaching affixes to them in a process known as inflection. 
# Consider the word JUMP. You can add affixes to it and form new words like 
# JUMPS, JUMPED, and JUMPING. In this case, the base word JUMP is the word stem.

def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

simple_stemmer("My system keeps crashing his crashed yesterday, ours crashes daily")

#Lemmitization
#the base form in this case is known as the root word, but not the root stem. The 
#difference being that the root word is always a lexicographically correct word 
#(present in the dictionary), but the root stem may not be so. Thus, root word, 
#also known as the lemma

def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

lemmatize_text("My system keeps crashing! his crashed yesterday, ours crashes daily")
