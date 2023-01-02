import torch
from typing import List
from collections import Counter
import tqdm
from tqdm import trange
import pickle5 as pickle 
from python_code.utility import import_baroni, cosine_similarity, create_context_dict, text_preprocessing, addDiagonal, create_combined_subset
import numpy as np

import fasttext
import fasttext.util
import datasets

from sklearn.metrics import average_precision_score
from transformers import (DistilBertTokenizerFast, DistilBertModel)


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
        
    
    def fit(self, words_of_interest):
        self._context_dict = {i : list() for i in set(words_of_interest)}

         # Creating a dictionary entry for each word in the texts

    def _update(self, all_text, words_of_interest, window = 1):

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


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    max_length = 50
    batch_size = 400
    unk_thresh = 10
    window = 5

    neg_file = "../Data_Shared/eacl2012-data/negative-examples.txtinput"
    pos_file = "../Data_Shared/eacl2012-data/positive-examples.txtinput"
    results_neg_file, results_pos_file, baroni, baroni_set = import_baroni(neg_file, pos_file)
    
    tok = Tokenizer()
    vocab = Context_dict()
    vocab.fit(tok.words(wikidata), baroni)
    
    ft = fasttext.load_model("../Data/ft_reduced_100.bin")
    
    # open pre processed wiki data
    # with open('../Data_Shared/wiki_subtext_preprocess.pickle', 'rb') as f:
    #         wiki_all_text = pickle.load(f)

    wikidata = datasets.load_dataset('wikipedia', '20200501.en')
    wikidata = wikidata['train']['text'][5000:10000]

    # Make a max scentence size
    wikidata = [sentence[:max_length].strip() if len(sentence.split()) > max_length else sentence.strip()
            for seq in tqdm(wikidata)
            for sentence in seq.split(".")]
    
    # Calculate number of batches 
    n_batches = 1 + (len(wikidata[:]) // batch_size)
    
    for k in trange(n_batches):
        # grab a batch_size chunk from seqs (wiki data)
        seqb = wikidata[batch_size*k:batch_size*(k+1)]
        Context_dict._update(seqb, window)

    for k in trange(n_batches):
        # grab a batch_size chunk from seqs (wiki data)
        seqb = wikidata[batch_size*k:batch_size*(k+1)]
        covariance = calculate_covariance(Context_dict._context_dict, ft, window)





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
