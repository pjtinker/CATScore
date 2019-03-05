# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 17:15:01 2018

@author: Josh Tinker
"""
import os
import random
import re

import numpy as np 

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

cList = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}

c_re = re.compile('(%s)' % '|'.join(cList.keys()))

# stop_words = set(stopwords.words('english'))

def expandContractions(text, c_re=c_re):
    """Expands common contractions into individual words.

    # Arguments
        text: list, text data.
        c_re: compiled regular expression to match.
    # Returns
        List of text with common contractions expanded
    """
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text)

def get_avg_words_per_sample(sample_texts):
    """Returns the median number of words per sample in given a corpus.
    # Arguments
    """
    total_words = 0
    for text in sample_texts:
        total_words += len(text)

    return float(total_words / len(sample_texts))

def get_num_words_per_sample(sample_texts):
    """Gets the median number of words per sample given corpus.
    # Arguments
        sample_texts: list, sample texts.
    # Returns
        float, median number of words per sample.
    """
    # num_words = []
    # for text in sample_texts:
    #     num_words.append(len(text.split()))
    return [len(s.split()) for s in sample_texts]

def processText(text, case=True, punct=True, expand=False, swords=False, lemmatize=False):
    """Removes unwanted/unnecessary characters from textual input.
    Removes all single and double quotes, removes all non-ascii characters,
    pad punctuation on each side if it remains.
    # Arguments
        text: list, text data.
        case: boolean; if true,  convert all words to lower
                    case, strip leading and trailing whitespace, pad punctuation on both 
					sides.
        swords: boolean; remove stopwords if true
		lemmatize: boolean; if true, lemmatize samples using nltk's WordNetLemmatizer
    # Returns
        List of processed text
    """
    #~ Attempt to strip out any non-ascii characters from the text.  
    text = re.sub(r"[^\x00-\x7F]+", ' ', text)
    #~ Attempt to remove any newline or tab characters
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')

    #~ Replace any opening character that is not a number or letter with a space.
    text = re.sub(r"^[^A-Za-z0-9]", ' ', text)  
    #~ Pad punctuation with spaces on both sides
    text = re.sub(r"([\.\",\(\)!\?;:/])", " \\1 ", text)
        
    if case:
        text = text.strip().lower()
        
    if expand:
        text = [expandContractions(x) for x in text]
        
    if swords:
        stop_words = set(stopwords.words('english'))
        text =  ' '.join([word for word in text.split() if word not in stop_words])

    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() ])
    #~ If any quotes remain, remove them.
    text = text.replace('"', '')
    text = text.replace("'", '')
    return text