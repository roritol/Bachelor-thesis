from typing import List
from collections import Counter
from tqdm import trange
from tqdm import tqdm
import pickle5 as pickle 
import datasets
import torch
import pandas as pd
from sklearn.metrics import average_precision_score

import fasttext
import fasttext.util

import sys

from transformers import (DistilBertTokenizerFast, DistilBertModel)
import re
import numpy as np  
from tqdm import tqdm
from numpy.linalg import norm
from scipy.spatial import distance
import torch
from collections import Counter


def import_baroni(neg_file, pos_file):
    results_neg_file, results_pos_file = [], []
    with open(neg_file) as f:
        for line in f:
            results_neg_file.append(line.replace("-n", "").replace("\n", "").strip("").split("\t"))
    with open(pos_file) as f:
        for line in f:
            results_pos_file.append(line.replace("-n", "").replace("\n", "").strip("").split("\t"))
    baroni = sum(results_neg_file, []) + sum(results_pos_file, [])
    baroni_set = set(baroni)

    return results_neg_file, results_pos_file, baroni, baroni_set

def import_hyperlex(file):
    results = []
    with open(file) as f:
        line = f.readline()
        while line:
            results.append(line.strip().split(" "))
            line = f.readline()
    f.close()

    return results


def text_preprocessing(
    text:list,
    punctuations = r'''!()-[]{};:'"\,<>./?@#$%^&*_“~''',
    stop_words=['and', 'a', 'is', 'the', 'in', 'be', 'will']
    )->list:
    """
    A method to preproces text
    """
    for x in text.lower(): 
        if x in punctuations: 
            text = text.replace(x, " ")

    # Removing words that have numbers in them
    text = re.sub(r'\w*\d\w*', '', text)

    # Removing digits
    text = re.sub(r'[0-9]+', '', text)

    # Cleaning the whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Setting every word to lower
    text = text.lower()

    # Converting all our text to a list 
#     text = text.split(' ')

    # Droping empty strings
#     text = [x for x in text if x!='']

    # Droping stop words
#     text = [x for x in text if x not in stop_words]

    return text

def cosine_similarity(a, b):
    nominator = np.dot(a, b)
    
    a_norm = np.sqrt(np.sum(a**2))
    b_norm = np.sqrt(np.sum(b**2))
    
    denominator = a_norm * b_norm
    
    cosine_similarity = nominator / denominator
    
    return cosine_similarity


def calculate_kl_bert(wordpair, embavg, is_diagonal, vocab):
    # Get the mean vectors and covariance matrices for the two words in the word pair
    mean1, covariance_matrix1 = embavg.get_mean_covariance(vocab._tok_to_id.get(wordpair[0])) 
    mean2, covariance_matrix2 = embavg.get_mean_covariance(vocab._tok_to_id.get(wordpair[1])) 
    
   
    # Create PyTorch multivariate normal distributions using the mean vectors and covariance matrices
    if bool(is_diagonal) is True:
        p = torch.distributions.multivariate_normal.MultivariateNormal(mean1, covariance_matrix=torch.diagflat(torch.diag(covariance_matrix1)))
        q = torch.distributions.multivariate_normal.MultivariateNormal(mean2, covariance_matrix=torch.diagflat(torch.diag(covariance_matrix2)))
    if bool(is_diagonal) is False:
        covarariance = torch.eye(len(covariance_matrix1))
        p = torch.distributions.multivariate_normal.MultivariateNormal(mean1, covarariance)
        q = torch.distributions.multivariate_normal.MultivariateNormal(mean2, covarariance)
        

    # Calculate the KL divergence between the two distributions
    kl = torch.distributions.kl.kl_divergence(p, q)

    return kl.item()


def bert_cosine_similarity(a, b):
    nominator = torch.dot(a, b)
    
    a_norm = torch.sqrt(torch.sum(a**2))
    b_norm = torch.sqrt(torch.sum(b**2))
    
    denominator = a_norm * b_norm
    
    cosine_similarity = nominator / denominator
    
    return cosine_similarity

#  In practice, it is also necessary to add a small ridge term 
#  δ > 0 to the diagonal of the matrix to regularize and avoid 
#  numerical problems when inverting - vilnis mccalumn

def addDiagonal(matrix, x):
    assert x < 1, f"x greater than 0 expected, got: {x}"
    
    for i in range(len(matrix)):
        matrix[i][i] = matrix[i][i] + x
    
    return matrix


def create_combined_subset(results_neg_file, results_pos_file, vocab):
    baroni_pos_subset = [x for x in results_pos_file if x[0] in vocab._tok_counts and x[1] in vocab._tok_counts]
    baroni_neg_subset = [x for x in results_neg_file if x[0] in vocab._tok_counts and x[1] in vocab._tok_counts]

    baroni_subset_label = []
    
    for i in baroni_pos_subset:
        baroni_subset_label.append([i, 1])

    for i in baroni_neg_subset:
        baroni_subset_label.append([i, 0])

    return baroni_pos_subset, baroni_neg_subset, baroni_subset_label


def calculate_kl(covariance, ft, wordpair):
    mean1 = torch.from_numpy(ft.get_word_vector(wordpair[0]))
    covariance_matrix1 = torch.from_numpy(covariance[wordpair[0]])
    covariance_matrix1 = addDiagonal(covariance_matrix1, 0.1)
    mean2 = torch.from_numpy(ft.get_word_vector(wordpair[1]))
    covariance_matrix2 = torch.from_numpy(covariance[wordpair[1]])
    covariance_matrix2 = addDiagonal(covariance_matrix2, 0.1)

    p = torch.distributions.multivariate_normal.MultivariateNormal(mean1, covariance_matrix=covariance_matrix1)
    q = torch.distributions.multivariate_normal.MultivariateNormal(mean2, covariance_matrix=covariance_matrix2)

    return float(torch.distributions.kl.kl_divergence(p, q))

# For the Emperical method

# def calculate_covariance(context_dict, ft, window):
#     covariance = {}

#     for word, context in context_dict.items():
#         total = torch.zeros((100,100))

#         for c_word in tqdm(context):
#             # would it be faster to store the matrixes of the words?
#             total += torch.from_numpy(np.outer((ft.get_word_vector(c_word) - 
#                                       ft.get_word_vector(word)), 
#                                       (ft.get_word_vector(c_word) - 
#                                       ft.get_word_vector(word))))
            
#             cov = (total / (len(context_dict[word]) * window))
#             covariance[word] = .001 * torch.eye(100) + cov

#     return covariance

def calculate_covariance(context_dict, ft, window):
    covariance = {}
    word_vectors = {}  # Store the word vectors of the words in the context_dict in memory
    
    for word in context_dict.keys():
        word_vectors[word] = ft.get_word_vector(word)
    
    for word, context in tqdm(context_dict.items()):
        total = torch.zeros((100,100))
        
        for c_word in context:
            if c_word not in word_vectors:  # If the word vector has not been stored yet, get it
                word_vectors[c_word] = ft.get_word_vector(c_word)
            
            total += torch.from_numpy(np.outer((word_vectors[c_word] - word_vectors[word]), 
                                      (word_vectors[c_word] - word_vectors[word])))
            
        cov = (total / (len(context_dict[word]) * window))
        covariance[word] = .001 * torch.eye(100) + cov
    return covariance


def calculate_kl_emp(covariance, ft, wordpair, is_diagonal):
    mean1 = torch.from_numpy(ft.get_word_vector(wordpair[0]))
    covariance_matrix1 = covariance[wordpair[0]]
    # covariance_matrix1 = addDiagonal(covariance_matrix1, 0.1)
    mean2 = torch.from_numpy(ft.get_word_vector(wordpair[1]))
    covariance_matrix2 = covariance[wordpair[1]]
    # covariance_matrix2 = addDiagonal(covariance_matrix2, 0.1)

    if is_diagonal:
        p = torch.distributions.multivariate_normal.MultivariateNormal(mean1, covariance_matrix=torch.diagflat(torch.diag(covariance_matrix1)))
        q = torch.distributions.multivariate_normal.MultivariateNormal(mean2, covariance_matrix=torch.diagflat(torch.diag(covariance_matrix2)))
    else:
        covarariance = torch.eye(len(covariance_matrix1))
        p = torch.distributions.multivariate_normal.MultivariateNormal(mean1, covarariance)
        q = torch.distributions.multivariate_normal.MultivariateNormal(mean2, covarariance)
        
    return float(torch.distributions.kl.kl_divergence(p, q))


class Vocab:

    def __init__(self):
        self._tok_counts = Counter()
        self._id_to_tok = {}

    def fit(self, data, word_list):
        for sequence in tqdm(data):
            self._tok_counts.update([tok for tok in sequence if tok in word_list])

        self._toks = (["</s>", "<unk>"] +
                      [tok for tok, _ in self._tok_counts.most_common()])
        self._tok_to_id = {tok: i for i, tok in enumerate(self._toks)}

    def __len__(self):
        return len(self._toks)


class EmbedAverages(torch.nn.Module):
    def __init__(self, n_words, dim):
        super().__init__()
        # matrix of wordvector sums
        self.register_buffer("_sum", torch.zeros(n_words, dim))
        self.register_buffer("_counts", torch.zeros(n_words, dtype=torch.long))
        self.register_buffer("_cov", torch.zeros(n_words, dim, dim))
    
    def add(self, ix, vec):
        self._counts[ix] += 1
        self._sum[ix] += vec
        self._cov[ix] += vec.reshape([len(vec), 1]) @ vec.reshape([1, len(vec)])
    
    def get_mean_covariance(self, ix):
        mean = self._sum[ix] / self._counts[ix]
        d = len(mean)
        cov = self._cov[ix] / self._counts[ix] - mean.reshape([d, 1])  @ mean.reshape([1, d])
        cov = .001 * torch.eye(d) + cov
        return mean, cov


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
        for i, word in enumerate(all_text):
            # Only update the context dict for words of interest
            if word in words_of_interest:
                for w in range(window):
                    # Getting the context that is ahead by *window* words
                    if i + 1 + w < len(all_text):
                        self._context_dict[word].append(all_text[(i + 1 + w)]) 
                    # Getting the context that is behind by *window* words    
                    if i - w - 1 >= 0:
                        self._context_dict[word].append(all_text[(i - w - 1)])


def main():
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    is_diagonal = bool(int(sys.argv[1]))
    max_context = int(sys.argv[2])
    use_curated_data = bool(int(sys.argv[3]))
    
    save_vocab = False
    batch_size = 200
    unk_thresh = 2
    max_length = 40
    # set the slice of the wikidata
    begin = 0
    end = 100    

    window = 5

    neg_file = "../data_shared/eacl2012-data/negative-examples.txtinput"
    pos_file = "../data_shared/eacl2012-data/positive-examples.txtinput"
    results_neg_file, results_pos_file, baroni, baroni_set = import_baroni(neg_file, pos_file)
    
   
    for j in tqdm(range(1,6)):
        for i in tqdm([0]):
            df = pd.DataFrame(columns =['Max Context', 'KL Score AP', 'COS Score AP'])
            max_context = i 

            if use_curated_data:
                print("open curated data:")
                with open(f'../data_shared/fixed/ramdom_curated0-25/curated{max_context}num{j}.pickle', 'rb') as f:
                    wikidata = pickle.load(f)
            else:
                wikidata = datasets.load_dataset('wikipedia', '20200501.en')
                wikidata = wikidata['train']['text'][int(begin):int(end)]
                print("truncating the scentences")
                wikidata = [sentence[:max_length].strip() if len(sentence.split()) > max_length else sentence.strip()
                    for seq in tqdm(wikidata)
                    for sentence in seq.split(".")]


            tok = Tokenizer()
            vocab = Vocab()
            print("fitting the vocab")
            vocab.fit(tok.words(baroni), baroni)


        # BERT METHOD

            if save_vocab:
                with open(f'../data_distrembed/curated1-10/vocab_is_diagonal_{is_diagonal}{i}.pickle', 'wb') as f:
                    pickle.dump(vocab,f)

            embavg = EmbedAverages(len(vocab), dim=768)
            model = DistilBertModel.from_pretrained("distilbert-base-uncased")
            model.to(device=device)

            n_batches = 1 + (len(baroni[:]) // batch_size)

            # no_grad() turns off the requirement of gradients by the tensor output (reduce memory usage)
            with torch.no_grad():
                for k in trange(n_batches):
                    # grab a batch_size chunk from seqs (wiki data)
                    seqb = baroni
                    # tokenize the batch to list of lists containing scentences, feed to bert, add last hidden state to embs
                    words, subw = tok(seqb)     # tokenizing the entire batch so scentences come to be stacked
                    mbart_input = subw.convert_to_tensors("pt").to(device=device)
                    out = model(**mbart_input, return_dict=True)
                    embs = out['last_hidden_state'].to(device='cpu')

                    for b in range(len(seqb)):
                        # accumulate eos token
                        for i, w in enumerate(words[b]):
                            span = subw.word_to_tokens(b, i)
                            if span is None:
                                continue
                            
                            if w not in vocab._tok_to_id:
                                continue

                            vec = embs[b, span.start]
                            embavg.add(vocab._tok_to_id[w], vec)

                torch.cuda.empty_cache()
            
            if save_vocab:
                torch.save(embavg, f"../data_distrembed/is_diagonal_{is_diagonal}{i}_vocab.embavg.pt")
                # embavg = torch.load('../data_distrembed/first10.avgs.pt')


        # EMPIRICAL METHOD


            context_dict = Context_dict()
            context_dict.fit(tok.words(baroni), baroni)

            ft = fasttext.load_model("../data/cc.en.100.bin")

            # Calculate number of batches 
            # n_batches = 1 + (len(wikidata[:]) // batch_size)
            

            # grab a batch_size chunk from seqs (wiki data)
            seqb = baroni
            words, _ = tok(seqb)
            all_text = [word for sentence in words for word in sentence]
            context_dict._update(all_text, baroni, window)

            covariance = calculate_covariance(context_dict._context_dict, ft, window)


        # COMBINE METHODS IN DATAFRAME


            # get true label in a list for neg and pos files 
            baroni_pos, baroni_neg, baroni_label = create_combined_subset(results_neg_file, results_pos_file, context_dict)
            
            # MAKE DATAFRAME
            df1 = pd.DataFrame(baroni_label, columns =['Wordpair', 'True label'])

            # CALCULATE KL and COS
            bert_kl = []
            bert_cos = []
            emp_kl = []
            emp_cos = []

            print("CALCULATE KL and COS")
            for wordpair in tqdm((baroni_pos + baroni_neg)):
                bert_kl.append(calculate_kl_bert(wordpair, embavg, is_diagonal, vocab))
                bert_cos.append(bert_cosine_similarity(embavg._sum[vocab._tok_to_id.get(wordpair[0])], 
                                                        embavg._sum[vocab._tok_to_id.get(wordpair[1])]))

                emp_kl.append(calculate_kl_emp(covariance, ft, wordpair, is_diagonal))
                emp_cos.append(cosine_similarity(ft.get_word_vector(wordpair[0]), 
                                                        ft.get_word_vector(wordpair[1])))


            df1['bert KL score'] = bert_kl
            df1['bert COS score'] = bert_cos
            df1['empirical KL score'] = emp_kl
            df1['empirical COS score'] = emp_cos


            with open(f'../data_shared/fixed/df_curated{max_context}_diag_{is_diagonal}num{j}.pickle', 'wb') as handle:
                pickle.dump(df1, handle, protocol=pickle.HIGHEST_PROTOCOL)

            print(df1)
            print("Diagonal             : ", is_diagonal)
            print("----------BERT RESULTS-----------")
            print("COS AP               : ", average_precision_score(df1["True label"], df1["bert COS score"]))
            print("KL AP                : ", average_precision_score(df1["True label"], -df1["bert KL score"]))
            print("--------EMPIRICAL RESULTS---------")
            print("COS AP               : ", average_precision_score(df1["True label"], df1["empirical COS score"]))
            print("KL AP                : ", average_precision_score(df1["True label"], -df1["empirical KL score"]))
            print("--------------STATS---------------")
            print("batch size           : ", batch_size)
            print("unkown threshold     : ", unk_thresh)
            print("context sentence     : ", max_context)
            print("Max scentence length : ", max_length)
            print(f"Wiki articles from  : {begin} to: {end}")
            print("total scentences     : ", len(wikidata))
            print("lowest vocab         : ", vocab._tok_counts.most_common()[-1])
            
            list1 = [f'BERT{max_context}', average_precision_score(df1["True label"], -df1["bert KL score"]), average_precision_score(df1["True label"], df1["bert COS score"])]
            list2 = [f'EMP{max_context}',  average_precision_score(df1["True label"], -df1["empirical KL score"]), average_precision_score(df1["True label"], df1["empirical COS score"])]
            df = pd.DataFrame([list1, list2],columns =['Max Context', 'KL Score AP', 'COS Score AP'])

            with open(f'../data_shared/fixed/df_AP{max_context}_{is_diagonal}num{j}.pickle', 'wb') as f:
                pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)

        






if __name__ == '__main__':
    main()
