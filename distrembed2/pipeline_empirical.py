import torch
from typing import List
from collections import Counter
from tqdm import trange
import pickle5 as pickle 

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

    def fit(self, data):
        for sequence in data:
            self._tok_counts.update(sequence)

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
        self.register_buffer("_ssq", torch.zeros(n_words, dim))
        self.register_buffer("_sum_normed", torch.zeros(n_words, dim))
        self.register_buffer("_counts", torch.zeros(n_words, dtype=torch.long))

    def add(self, ix, vec):
        # could use B.index_add(0, ix, torch.ones_like(ix, dtype=torch.float)
        self._counts[ix] += 1
        self._sum[ix] += vec
        self._ssq[ix] += vec ** 2
        # self._sum_normed[ix] += vec / torch.norm(vec, dim=-1, keepdim=True)

# after you made the lookup tabel you have to take the row vector from sum matrix 
# corrsponding to the taget word and devide it by the count 

# vocab wil give index of target word 

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


def main():
    neg_file = "../Data_Shared/eacl2012-data/negative-examples.txtinput"
    pos_file = "../Data_Shared/eacl2012-data/positive-examples.txtinput"
    results_neg_file, results_pos_file, baroni, baroni_set = import_baroni(neg_file, pos_file)

    # Load the "embavg" file saved at the end of the script
    embavg = torch.load('../data_distrembed/roen.avgs.pt')

    seqs = baroni
    vocab = Vocab()
    tok = Tokenizer()
    vocab.fit(tok.words(seqs))

    # Create a dictionary that maps words to their vector representations
    word_vectors = {}

    # Iterate over the words in the vocabulary
    for key, id in vocab._tok_to_id.items():
        # Get the average vector for the current word
        vec = embavg._sum[id]
        count = embavg._counts[id]
        # Add an entry to the dictionary that maps the current word to its average vector
        word_vectors[key] = vec.numpy() / count.numpy()
    
    print(word_vectors)
    # with open('covariance_BERT.pickle', 'wb') as handle:
    #     pickle.dump(word_vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
