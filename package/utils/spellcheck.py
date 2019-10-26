import io
import sys
import pandas as pd
import numpy as np
import operator 
import re
from collections import Counter
import gensim
import heapq
from multiprocessing import Pool
from tqdm import tqdm
tqdm.pandas()

# from preprocess_text import processText

EMBEDDING_PATH = 'D:/Research/TextClassification/glove6b/glove.6B.100d.txt'

class SpellCheck():
    def __init__(self, text_data = [], top_k=90000):
        self.text_data = text_data
        self.top_k = top_k
        self.w_rank = {}
        self.vocab = {}
        self.misspelt_dict = {}
        self.load_embedding_vocab()
        self.generate_misspell_dict()

    def words(self, text): return re.findall(r'\w+', text.lower())

    def load_embedding_vocab(self):
        # TODO: Configure to use with embeddings after final dir structure known
        # TODO: Save misspelt dict and make it availalbe to be saved/loaded by this class
        #  we can do one spellcheck, generate the dict, then reuse for future spellchecking
        words = []
        try:
            print("Loading embeddding vocabulary...")
            f = open(EMBEDDING_PATH, encoding='utf-8')
            for line in tqdm(f):
                values = line.split()
                words.append(values[0])    
            f.close()

            for i,word in enumerate(words):
                self.w_rank[word] = i

        except IOError as ioe:
            print("Unable to open embedding file.")
            print(ioe)
            sys.exit(1)

    def P(self, word): 
        # "Probability of `word`."
        # use inverse of rank as proxy
        # returns 0 if the word isn't in the dictionary
        return - self.w_rank.get(word, 0)

    def correction(self, word): 
        "Most probable spelling correction for word."
        return max(self.candidates(word), key=self.P)

    def candidates(self, word): 
        "Generate possible spelling corrections for word."
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])

    def known(self, words): 
        "The subset of `words` that appear in the dictionary of self.w_rank."
        return set(w for w in words if w in self.w_rank)

    def edits1(self, word):
        # "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word): 
        # "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def build_vocab(self, texts):
        sentences = [str(x).split() for x in texts]
        print("Building vocabulary from data...")
        for sentence in tqdm(sentences):
            for word in sentence:
                try:
                    self.vocab[word] += 1
                except KeyError:
                    self.vocab[word] = 1     

    def generate_misspell_dict(self):
        self.build_vocab(self.text_data)
        top_k_words = dict(heapq.nlargest(self.top_k, self.vocab.items(), key=operator.itemgetter(1)))
        pool = Pool(4)
        corrected_words = pool.map(self.correction, list(top_k_words.keys()))
        print("Generating detected misspellings/OOV...")
        for word, corrected_word in tqdm(zip(top_k_words, corrected_words)):
            if word!=corrected_word:
                # print(word,":",corrected_word)
                self.misspelt_dict[word] = corrected_word
        print("Found potential corrections for {} words".format(len(self.misspelt_dict)))
        try:
            self.misspell_re = re.compile('(%s)' % '|'.join(self.misspelt_dict.keys()))
        except re.error as rerror:
            print("Vocabulary contained invalid characters.  Did you remove punctuation?")
            return
        except Exception as e:
            print("Exception occured: {}".format(e))
            return
        # self.misspell_re = re.compile('(%s)' % '|'.join(self.misspelt_dict.keys()))


    def correct_spelling(self, text):
        """Attempts to correct misspellings via comparison to GLoVe embeddings and 
        Norvig's spell checker.
        # Arguments
            text: String, sample sentence to correct.
        """
        def replace(match):
            return self.misspelt_dict[match.group(0)]

        return self.misspell_re.sub(replace, str(text))
        
    def get_vocab(self):
        return self.vocab

    def get_misspell_dict(self):
        return self.misspelt_dict

if __name__ == '__main__':
    data = pd.read_csv('D:/Utilities/testbed/flair/q6/train.csv', sep='\t', quoting=1, header=None)
    # processed = data[1].progress_apply(lambda x: processText(str(x)))
    # sentences = processed.progress_apply(lambda x: str(x).split()).values
    sentences = data[1].progress_apply(lambda x: str(x).split()).values
    sc = SpellCheck(sentences, 90000)


