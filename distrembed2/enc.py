from typing import List
from collections import Counter
from tqdm import trange
import pickle5 as pickle 
import datasets
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import average_precision_score

import sys

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


def calculate_kl(wordpair, embavg, vocab):
    # Get the mean vectors and covariance matrices for the two words in the word pair
    mean1, covariance_matrix1 = embavg.get_mean_covariance(vocab._tok_to_id.get(wordpair[0])) 
    mean2, covariance_matrix2 = embavg.get_mean_covariance(vocab._tok_to_id.get(wordpair[1])) 
    
    # Create PyTorch multivariate normal distributions using the mean vectors and covariance matrices
    p = torch.distributions.multivariate_normal.MultivariateNormal(mean1, covariance_matrix=covariance_matrix1)
    q = torch.distributions.multivariate_normal.MultivariateNormal(mean2, covariance_matrix=covariance_matrix2)

    # Calculate the KL divergence between the two distributions
    kl = torch.distributions.kl.kl_divergence(p, q)

    return kl.item()

def calculate_diag_kl(wordpair, embavg, vocab):
    
    # Get the mean vectors and covariance matrices for the two words in the word pair
    mean1, covariance_matrix1 = embavg.get_mean_covariance(vocab._tok_to_id.get(wordpair[0])) 
    mean2, covariance_matrix2 = embavg.get_mean_covariance(vocab._tok_to_id.get(wordpair[1])) 
    n = int(100)
    # Create PyTorch multivariate normal distributions using the mean vectors and covariance matrices
    p = torch.distributions.multivariate_normal.MultivariateNormal(mean1, covariance_matrix=torch.diagflat(torch.diag(covariance_matrix1)))
    q = torch.distributions.multivariate_normal.MultivariateNormal(mean2, covariance_matrix=torch.diagflat(torch.diag(covariance_matrix2)))

    # Calculate the KL divergence between the two distributions
    kl = torch.distributions.kl.kl_divergence(p, q)

    return kl.item()


def cosine_similarity(a, b):
    nominator = torch.dot(a, b)
    
    a_norm = torch.sqrt(torch.sum(a**2))
    b_norm = torch.sqrt(torch.sum(b**2))
    
    denominator = a_norm * b_norm
    
    cosine_similarity = nominator / denominator
    
    return cosine_similarity

def diag_cosine_similarity(a, b):
    a = torch.diag_embed(torch.diagonal(a))
    b = torch.diag_embed(torch.diagonal(b))
    nominator = torch.mm(a, b)
    
    a_norm = torch.sqrt(torch.sum(a**2))
    b_norm = torch.sqrt(torch.sum(b**2))
    
    denominator = a_norm * b_norm
    
    cosine_similarity = nominator / denominator
    
    return cosine_similarity


def main():
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # print('cmd entry:', sys.argv)
    # is_diagonal = eval(sys.argv[0])
    # # assert isinstance(is_diagonal, bool)
    # # raise TypeError('param should be a bool')
    is_diagonal = False
    batch_size = 200
    unk_thresh = 2
    max_length = 40
    # set the slice of the wikidata
    begin = 50000
    end = 51000


    neg_file = "../Data_Shared/eacl2012-data/negative-examples.txtinput"
    pos_file = "../Data_Shared/eacl2012-data/positive-examples.txtinput"
    results_neg_file, results_pos_file, baroni, baroni_set = import_baroni(neg_file, pos_file)
    
    wikidata = datasets.load_dataset('wikipedia', '20200501.en')
    wikidata = wikidata['train']['text'][int(begin):int(end)]

    # print("open pickeled data:")

    # with open('../Python_Code/wiki_preprocessed1.pickle', 'rb') as f:
    #     wikidata = pickle.load(f)
    


    print("truncating the scentences")
    wikidata = [sentence[:max_length].strip() if len(sentence.split()) > max_length else sentence.strip()
            for seq in tqdm(wikidata)
            for sentence in seq.split(".")]

    tok = Tokenizer()
    vocab = Vocab()
    print("fitting the vocab")
    vocab.fit(tok.words(wikidata), baroni)

    with open('../data_distrembed/onetenth_vocab.pickle', 'wb') as f:
        pickle.dump(vocab,f)

    print("--------------------------")
    print("--------------------------")
    print("--------------------------")

    with open('../Data/vocab:1000pickle', 'wb') as f:
        pickle.dump(vocab,f)

    embavg = EmbedAverages(len(vocab), dim=768)
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model.to(device=device)

    n_batches = 1 + (len(wikidata[:]) // batch_size)

    # no_grad() turns off the requirement of gradients by the tensor output (reduce memory usage)
    with torch.no_grad():
        for k in trange(n_batches):
            # grab a batch_size chunk from seqs (wiki data)
            seqb = wikidata[batch_size*k:batch_size*(k+1)]
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

                #     if vocab._tok_counts[w] < unk_thresh:
                #         embavg.add(vocab._tok_to_id["<unk>"], vec)

                # if span is not None:
                #     eos_ix = span.end
                #     embavg.add(vocab._tok_to_id["</s>"], embs[b, eos_ix])

        torch.cuda.empty_cache()

    print("open the embavg file")
    torch.save(embavg, "../data_distrembed/first10.embavg.pt")
    # embavg = torch.load('../data_distrembed/first10.avgs.pt')
    # get f1 scores etc

    print("make subsets")
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
    print("CALCULATE KL and COS")

    if is_diagonal:
        for wordpair in tqdm((baroni_pos_subset + baroni_neg_subset)):
            baroni_subset_kl.append(calculate_diag_kl(wordpair, embavg, vocab))
            baroni_subset_cos.append(diag_cosine_similarity(embavg._sum[vocab._tok_to_id.get(wordpair[0])], 
                                                    embavg._sum[vocab._tok_to_id.get(wordpair[1])]))
    else:
        for wordpair in tqdm((baroni_pos_subset + baroni_neg_subset)):
            baroni_subset_kl.append(calculate_kl(wordpair, embavg, vocab))
            baroni_subset_cos.append(cosine_similarity(embavg._sum[vocab._tok_to_id.get(wordpair[0])], 
                                                    embavg._sum[vocab._tok_to_id.get(wordpair[1])]))

    df1['KL score'] = baroni_subset_kl
    df1['COS score'] = baroni_subset_cos

    # with open('df1.pickle', 'wb') as handle:
    #     pickle.dump(df1, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(df1)
    print("COS AP               : ", average_precision_score(df1["True label"], df1["COS score"]))
    print("KL AP                : ", average_precision_score(df1["True label"], -df1["KL score"]))
    print("batch size           : ", batch_size)
    print("unkown threshold     : ", unk_thresh)
    print("Max scentence length : ", max_length)
    print(f"Wiki articles from  : {begin} to: {end}")
    print("total scentences     : ", len(wikidata))
    print("lowest vocab         : ", vocab._tok_counts.most_common()[-1])



if __name__ == '__main__':
    main()
