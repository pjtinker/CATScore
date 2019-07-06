import io

import pandas as pd
import numpy as np
import operator 
from collections import Counter
from tqdm import tqdm
tqdm.pandas()

# from preprocess_text import processText
# from spellcheck import SpellCheck

EMBEDDING_PATH = 'D:/Research/TextClassification/glove6b/glove.6B.100d.txt'
FT_PATH = 'D:/Research/TextClassification/fasttextv2/crawl-300d-2M.vec'
TOP_K = 20000

class EmbeddingUtils():

    def __init__(self):
        self.embedding_matrix = None
        self.embedding_type = None
        self.embedding_dims = None
        self.vocab = {}
        self.oov_list = []

    def generate_embedding_matrix(self, word_index, embedding_type='glove', embedding_dims=100):
        self.embedding_type = embedding_type
        self.embedding_dims = embedding_dims
        # self.word_index = word_index
        if embedding_type == 'glove':
            embedding_matrix_full = self.get_embedding_matrix()
        elif embedding_type == 'word2vec':
            pass
        num_words = min(len(word_index) + 1, TOP_K)
        self.embedding_matrix = np.zeros((num_words, embedding_dims))
        print("Generating word embeddings...")
  
        for word, i in word_index.items():
            if i >= TOP_K:
                continue
            try:
                embedding_vector = embedding_matrix_full[word]
            except KeyError:
                self.oov_list.append(word)
                embedding_vector = None
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector
        print("Word embedding matrix generated!")
        print("Out of vocabulary word count: ", len(self.oov_list))

    def get_em(self):
        return self.embedding_matrix

    def get_embedding_matrix(self, path=EMBEDDING_PATH):
        print("Loading GloVe embeddings...")
        embeddings_index = {}
        f = open(path, encoding='utf-8')
        for line in tqdm(f):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        return embeddings_index

    def get_fasttext_embedding_matrix(self, path=FT_PATH):
        fin = io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        for line in tqdm(fin):
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = map(float, tokens[1:])
        return data

    def build_vocab(self, sentences, verbose =  True):
        """
        :param sentences: list of list of words
        :return: dictionary of words and their count
        """
        vocab = {}
        for sentence in tqdm(sentences, disable = (not verbose)):
            for word in sentence:
                try:
                    vocab[word] += 1
                except KeyError:
                    vocab[word] = 1
        return vocab

    def add_lower(self, embedding, vocab):
        count = 0
        for word in vocab:
            if word in embedding and word.lower() not in embedding:  
                embedding[word.lower()] = embedding[word]
                count += 1
        print(f"Added {count} words to embedding")

    def check_coverage(self, vocab, embeddings_index):
        a = {}
        oov = {}
        k = 0
        i = 0
        for word in tqdm(vocab):
            try:
                a[word] = embeddings_index[word]
                k += vocab[word]
            except:

                oov[word] = vocab[word]
                i += vocab[word]
                pass

        print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
        print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
        sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

        return sorted_x

if __name__ == '__main__':
    data = pd.read_csv('D:/Utilities/testbed/flair/q6/train.csv', sep='\t', quoting=1, header=None)
    sentences = data[1].progress_apply(lambda x: processText(str(x)))
    sentences = sentences.progress_apply(lambda x: str(x).split()).values
    vocab = build_vocab(sentences)
    print({k: vocab[k] for k in list(vocab)[:5]})

    embedding_index = get_embedding_matrix()
    # embedding_index = get_fasttext_embedding_matrix()
    oov = check_coverage(vocab, embedding_index)
    print("Length of oov: {}".format(len(oov)))
    print("Top 20 OOV: {}".format(oov[:20]))

    add_lower(vocab, embedding_index)

    oov = check_coverage(vocab, embedding_index)
    print("After adding lower case...")
    print("Length of oov: {}".format(len(oov)))
    print("Top 20 OOV: {}".format(oov[:20]))

    sc = SpellCheck(sentences, 90000)
    sentences = [sc.correct_spelling(x) for x in sentences]
    vocab = build_vocab(sentences)
    oov = check_coverage(vocab, embedding_index)
    print("After correcting spelling...")
    print("Length of oov: {}".format(len(oov)))
    print("Top 20 OOV: {}".format(oov[:20]))