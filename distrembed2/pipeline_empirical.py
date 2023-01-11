import torch
from typing import List
from collections import Counter

from tqdm import trange
from tqdm import tqdm
import pickle5 as pickle 
# from python_code.utility import import_baroni, cosine_similarity, addDiagonal
import numpy as np
import ast
import pandas as pd

import fasttext
import fasttext.util
import datasets

from sklearn.metrics import average_precision_score
from transformers import (DistilBertTokenizerFast, DistilBertModel)

def import_baroni(neg_file, pos_file):
    filenames = ["neg_file", "pos_file"]

    for i, file in enumerate([neg_file, pos_file]):
        globals()['results_{}'.format(filenames[i])] = []
        
        with open(file) as f:
            line = f.readline()
            while line:
                globals()['results_{}'.format(filenames[i])].append(line.replace("-n", "").replace("\n", "").strip("").split("\t"))
                line = f.readline()
        f.close()

    baroni = sum(results_neg_file, []) + sum(results_pos_file, [])
    baroni_set = set(baroni)

    return results_neg_file, results_pos_file, baroni, baroni_set

def cosine_similarity(a, b):
    nominator = np.dot(a, b)
    
    a_norm = np.sqrt(np.sum(a**2))
    b_norm = np.sqrt(np.sum(b**2))
    
    denominator = a_norm * b_norm
    
    cosine_similarity = nominator / denominator
    
    return cosine_similarity

def addDiagonal(matrix, x):
    assert x < 1, f"x greater than 0 expected, got: {x}"
    
    for i in range(len(matrix)):
        matrix[i][i] = matrix[i][i] + x
    
    return matrix

class Tokenizer:

    def __init__(self):
        self._t = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    
    def words(self, sequences: List[str]):
        return [s.split() for s in sequences]

    def __call__(self, sequences: List[str]):
        words = self.words(sequences)
        subw = self._t.batch_encode_plus(words,
                                         is_split_into_words=True,
                                         padding=True)
        return words, subw


class Context_dict:

    def __init__(self):
        self._tok_counts = Counter()
        self._context_dict = {}
        
    
    def fit(self, data, words_of_interest):
        self._context_dict = {i : list() for i in set(words_of_interest)}
        
        for sequence in tqdm(data):
            self._tok_counts.update([tok for tok in sequence if tok in words_of_interest])

         # Creating a dictionary entry for each word in the texts

    def _update(self, all_text, words_of_interest, window):
        for i, word in tqdm(enumerate(all_text)):
            # Only update the context dict for words of interest
            if word in words_of_interest:
                for w in range(window):
                    # Getting the context that is ahead by *window* words
                    if i + 1 + w < len(all_text):
                        self._context_dict[word].append(all_text[(i + 1 + w)]) 
                    # Getting the context that is behind by *window* words    
                    if i - w - 1 >= 0:
                        self._context_dict[word].append(all_text[(i - w - 1)])


def calculate_covariance(context_dict, ft, window):
    covariance = {}

    for word, context in tqdm(context_dict.items()):
        total = torch.zeros((100,100))

        for c_word in context:
            # would it be faster to store the matrixes of the words?
            total += torch.from_numpy(np.outer((ft.get_word_vector(c_word) - 
                                      ft.get_word_vector(word)), 
                                      (ft.get_word_vector(c_word) - 
                                      ft.get_word_vector(word))))
            
            cov = (total / (len(context_dict[word]) * window))
            covariance[word] = .001 * torch.eye(100) + cov

    return covariance

def calculate_kl(covariance, ft, wordpair):
    mean1 = torch.from_numpy(ft.get_word_vector(wordpair[0]))
    covariance_matrix1 = covariance[wordpair[0]]
    covariance_matrix1 = addDiagonal(covariance_matrix1, 0.1)
    mean2 = torch.from_numpy(ft.get_word_vector(wordpair[1]))
    covariance_matrix2 = covariance[wordpair[1]]
    covariance_matrix2 = addDiagonal(covariance_matrix2, 0.1)

    p = torch.distributions.multivariate_normal.MultivariateNormal(mean1, covariance_matrix=covariance_matrix1)
    q = torch.distributions.multivariate_normal.MultivariateNormal(mean2, covariance_matrix=covariance_matrix2)

    return float(torch.distributions.kl.kl_divergence(p, q))


def main():
    max_length = 50
    batch_size = 400
    # unk_thresh = 10
    window = 5

    neg_file = "../data_shared/eacl2012-data/negative-examples.txtinput"
    pos_file = "../data_shared/eacl2012-data/positive-examples.txtinput"
    results_neg_file, results_pos_file, baroni, baroni_set = import_baroni(neg_file, pos_file)
    
    # with open('../data_shared/wiki_subset.txt') as file:
    #     data = file.read()

    # wikidata = ast.literal_eval(data)
    # wikidata = wikidata["text"][:100]   

    # wikidata = [sentence[:max_length].strip() if len(sentence.split()) > max_length else sentence.strip()
    #         for seq in tqdm(wikidata)
    #         for sentence in seq.split(".")]

    with open('../data_distrembed/curated50000.pickle', 'rb') as f:
        wikidata = pickle.load(f)

    tok = Tokenizer()
    vocab = Context_dict()
    vocab.fit(tok.words(wikidata), baroni_set)
    
    ft = fasttext.load_model("../Data/cc.en.100.bin")
    
    # Calculate number of batches 
    n_batches = 1 + (len(wikidata[:]) // batch_size)
    
    for k in trange(n_batches):
        # grab a batch_size chunk from seqs (wiki data)
        seqb = wikidata[batch_size*k:batch_size*(k+1)]
        words, subwords = tok(seqb)
        all_text = [word for sentence in words for word in sentence]
        vocab._update(all_text, baroni_set, window)

    
    covariance = calculate_covariance(vocab._context_dict, ft, window)


    baroni_pos_subset = [x for x in results_pos_file if x[0] in vocab._tok_counts and x[1] in vocab._tok_counts]
    baroni_neg_subset = [x for x in results_neg_file if x[0] in vocab._tok_counts and x[1] in vocab._tok_counts]


    baroni_subset_label = []

    for i in baroni_pos_subset:
        baroni_subset_label.append([i, 1])

    for i in baroni_neg_subset:
        baroni_subset_label.append([i, 0])

    # MAKE DATAFRAME
    df1 = pd.DataFrame(baroni_subset_label, columns =['Wordpair', 'True label'])

    # CALCULATE KL and COS
    baroni_subset_kl = []
    baroni_subset_cos = []

    for wordpair in tqdm((baroni_pos_subset + baroni_neg_subset)):
        baroni_subset_kl.append(calculate_kl(covariance, ft, wordpair))
        baroni_subset_cos.append(cosine_similarity(ft.get_word_vector(wordpair[0]), 
                                                   ft.get_word_vector(wordpair[1])))
       
    df1['KL score'] = baroni_subset_kl
    df1['COS score'] = baroni_subset_cos

    with open('df1.pickle', 'wb') as handle:
        pickle.dump(df1, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("COS AP: ", average_precision_score(df1["True label"], df1["COS score"]))
    print("KL AP: ", average_precision_score(df1["True label"], df1["KL score"]))

if __name__ == '__main__':
    main()
