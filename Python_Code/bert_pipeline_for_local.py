import torch
from typing import List
from collections import Counter
import pickle5 as pickle 

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.metrics import average_precision_score
from utility import import_baroni
from transformers import (DistilBertTokenizerFast, DistilBertModel)

class Vocab:

    def __init__(self):
        self._tok_counts = Counter()
        self._id_to_tok = {}

    def fit(self, data):
        for sequence in data:
            self._tok_counts.update(sequence)

        self._toks = (["</s>", "<unk>"] +
                      [tok for tok, _ in self._tok_counts.most_common()])
        self._tok_to_id = {tok: i for i, tok in enumerate(self._toks)}
        self._id_to_tok = {i: tok for i, tok in enumerate(self._toks)}

    def __len__(self):
        return len(self._toks)

class EmbedAverages(torch.nn.Module):

    def __init__(self, n_words, dim):

        super().__init__()
        # matrix of wordvector sums
        self.register_buffer("_sum", torch.zeros(n_words, dim))
        self.register_buffer("_ssq", torch.zeros(n_words, dim))
        self.register_buffer("_sum_normed", torch.zeros(n_words, dim))
        self.register_buffer("_counts", torch.zeros(n_words, dtype=torch.long))

    def add(self, ix, vec):
        self._counts[ix] += 1
        self._sum[ix] += vec
        self._ssq[ix] += vec ** 2

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

def calculate_covariance(context_dict, baroni_set, ft, window):
    covariance = {}

    for word in tqdm(baroni_set):
        total = np.zeros((100,100))

        for c_word in context_dict[word]:
            total += np.outer((ft.get_word_vector(c_word) - ft.get_word_vector(word)), (ft.get_word_vector(c_word) - ft.get_word_vector(word)))

        covariance[word] = (total / (len(context_dict[word]) * window))

    return covariance

def create_word_covariance_matrix(embed_averages, i, word):
    # Extract the mean and variance tensors for the word at index i
    mean_tensor = embed_averages._sum[i] / embed_averages._counts[i]
    var_tensor = (embed_averages._ssq[i] - mean_tensor ** 2) / embed_averages._counts[i]

    # Stack the mean and variance tensors along the second dimension
    mean_var_tensor = torch.stack([mean_tensor, var_tensor], dim=1)

    # Calculate the covariance matrix for the word by taking the outer product of the mean_var_tensor
    # with itself
    cov_matrix = mean_var_tensor.t() @ mean_var_tensor

    return cov_matrix

def calculate_kl(covariance, wordpair):
    # Get the mean vectors and covariance matrices for the two words in the word pair
    mean1 = covariance.get(wordpair[0])[1]
    covariance_matrix1 = covariance.get(wordpair[0])[0]
    mean2 = covariance.get(wordpair[1])[1]
    covariance_matrix2 = covariance.get(wordpair[1])[0]

    # Create PyTorch multivariate normal distributions using the mean vectors and covariance matrices
    p = torch.distributions.multivariate_normal.MultivariateNormal(mean1, covariance_matrix=covariance_matrix1)
    q = torch.distributions.multivariate_normal.MultivariateNormal(mean2, covariance_matrix=covariance_matrix2)

    # Calculate the KL divergence between the two distributions
    kl = torch.distributions.kl.kl_divergence(p, q)

    return kl.item()


def main():
    
    neg_file = "../Data_Shared/eacl2012-data/negative-examples.txtinput"
    pos_file = "../Data_Shared/eacl2012-data/positive-examples.txtinput"
    results_neg_file, results_pos_file, baroni, baroni_set = import_baroni(neg_file, pos_file)

    # with open('../distrembed2/covariance_BERT.pickle', 'rb') as f:
    #         covariance_BERT = pickle.load(f)

    # print(len(covariance_BERT))
    # for key, value in covariance_BERT.items():
    #     if value.all() == 0:
    #         print(key)
    #         covariance_BERT.pop(key)

    # print(len(covariance_BERT))

    embavg = torch.load('../data_distrembed/roen.avgs.pt')

    seqs = baroni
    vocab = Vocab()
    tok = Tokenizer()
    vocab.fit(tok.words(seqs))
    
    # covariance = calculate_covariance(context_dict, combined_set, ft, window)
    # Calculate the covariance matrix for each word

    word_cov_matrices = {}
    for i in range(len(embavg._counts)):
        word = vocab._id_to_tok[i]
        word_cov_matrix = create_word_covariance_matrix(embavg, i, word)
        word_cov_matrices[word] = [word_cov_matrix, embavg._sum[i]]

    # baroni_pos_subset, baroni_neg_subset = create_combined_subset(word_cov_matrices, results_neg_file, results_pos_file, combined_set)

    baroni_subset_label = []

    for i in results_pos_file:
        baroni_subset_label.append([i, 1])

    for i in results_neg_file:
        baroni_subset_label.append([i, 0])

    # MAKE DATAFRAME
    df1 = pd.DataFrame(baroni_subset_label, columns =['Wordpair', 'True label'])

    # CALCULATE KL and COS
    baroni_subset_kl = []
    baroni_subset_cos = []

    for wordpair in tqdm((results_pos_file + results_neg_file)):
        baroni_subset_kl.append(calculate_kl(word_cov_matrices, wordpair))
        # baroni_subset_cos.append(cosine_similarity(ft.get_word_vector(wordpair[0]).numpy(), ft.get_word_vector(wordpair[1]).numpy()))
       
    df1['KL score'] = baroni_subset_kl
    df1['COS score'] = baroni_subset_cos

    with open('df1.pickle', 'wb') as handle:
        pickle.dump(df1, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(df1)
    # print("COS AP: ", average_precision_score(df1["True label"], df1["COS score"]))
    # print("KL AP: ", average_precision_score(df1["True label"], df1["KL score"]))

if __name__ == '__main__':
    main()

