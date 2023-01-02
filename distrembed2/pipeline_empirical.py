import torch
from typing import List
from collections import Counter
from tqdm import trange
import pickle5 as pickle 
from python_code.utility import import_baroni

from transformers import (DistilBertTokenizerFast, DistilBertModel)

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


def main():
    neg_file = "../Data_Shared/eacl2012-data/negative-examples.txtinput"
    pos_file = "../Data_Shared/eacl2012-data/positive-examples.txtinput"
    results_neg_file, results_pos_file, baroni, baroni_set = import_baroni(neg_file, pos_file)
    
    seqs = baroni
    vocab = Vocab()
    tok = Tokenizer()
    vocab.fit(tok.words(seqs))
    
    
    
    ft = fasttext.load_model("../Data/ft_reduced_100.bin")
    
    # open pre processed wiki data
    with open('../Data_Shared/wiki_subtext_preprocess.pickle', 'rb') as f:
            wiki_all_text = pickle.load(f)

    # creating a context dictionary
    print("create context dict")
    window = 5
    context_dict = create_context_dict(wiki_all_text, window)
    combined_set = set(wiki_all_text)&set(baroni_set)
    covariance = calculate_covariance(context_dict, combined_set, ft, window)
    baroni_pos_subset, baroni_neg_subset = create_combined_subset(covariance, results_neg_file, results_pos_file, combined_set)

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
