from typing import List
from collections import Counter
from tqdm import trange
import pickle5 as pickle 

import torch

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
        for sequence in data:
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


def calculate_kl(wordpair):
    # Get the mean vectors and covariance matrices for the two words in the word pair
    mean1, covariance_matrix1 = embavg.get_mean_covariance(vocab._tok_to_id.get(wordpair[0])) 
    mean2, covariance_matrix2 = embavg.get_mean_covariance(vocab._tok_to_id.get(wordpair[1])) 
    
    # Create PyTorch multivariate normal distributions using the mean vectors and covariance matrices
    p = torch.distributions.multivariate_normal.MultivariateNormal(mean1, covariance_matrix=covariance_matrix1)
    q = torch.distributions.multivariate_normal.MultivariateNormal(mean2, covariance_matrix=covariance_matrix2)

    # Calculate the KL divergence between the two distributions
    kl = torch.distributions.kl.kl_divergence(p, q)

    return kl.item()


def main():
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    batch_size = 128
    unk_thresh = 20

    neg_file = "../Data_Shared/eacl2012-data/negative-examples.txtinput"
    pos_file = "../Data_Shared/eacl2012-data/positive-examples.txtinput"
    results_neg_file, results_pos_file, baroni, baroni_set = import_baroni(neg_file, pos_file)

    # seqs has to become the wiki dataset 
    with open('../Data_Shared/wiki_subtext_preprocess.pickle', 'rb') as handle:
        seqs = pickle.load(handle)


    tok = Tokenizer()
    vocab = Vocab()
    vocab.fit(tok.words(seqs), baroni)
    print(vocab._tok_to_id.get("church"))

    with open("../data_distrembed/roen.vocab", "w") as f:
        for w in vocab._toks:
            print(w, file=f)


    embavg = EmbedAverages(len(vocab), dim=768)
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model.to(device=device)

    n_batches = 1 + (len(seqs[:]) // batch_size)

    # no_grad() turns off the requirement of gradients by the tensor output (reduce memory usage)
    with torch.no_grad():
        for k in trange(n_batches):
            # grab a batch_size chunk from seqs (wiki data)
            seqb = seqs[batch_size*k:batch_size*(k+1)]

            # tokenize the batch, feed to bert, add last hidden state to embs
            words, subw = tok(seqb)
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

                    if vocab._tok_counts[w] < unk_thresh:
                        embavg.add(vocab._tok_to_id["<unk>"], vec)

                if span is not None:
                    eos_ix = span.end
                    embavg.add(vocab._tok_to_id["</s>"], embs[b, eos_ix])


    
    torch.save(embavg, "../data_distrembed/first10.avgs.pt")
    

if __name__ == '__main__':
    main()

