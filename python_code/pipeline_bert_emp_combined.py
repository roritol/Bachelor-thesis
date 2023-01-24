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

from utility import create_combined_subset, calculate_kl_emp, calculate_covariance
from utility import import_baroni, cosine_similarity, calculate_kl_bert, bert_cosine_similarity

import sys

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
    
    save_vocab = True
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
    
    if use_curated_data:
        print("open curated data:")
        with open(f'../data_shared/fixed/curated{max_context}.pickle', 'rb') as f:
            wikidata = pickle.load(f)
    else:
        wikidata = datasets.load_dataset('wikipedia', '20200501.en')
        wikidata = wikidata['train']['text'][int(begin):int(end)]

        print("truncating the scentences")
        wikidata = [sentence[:max_length].strip() if len(sentence.split()) > max_length else sentence.strip()
                for seq in tqdm(wikidata)
                for sentence in seq.split(".")]

    # import ast
    # with open('../data_shared/wiki_subset.txt') as f:
    #     wikidata = f.read()
        
    # wikidata = ast.literal_eval(wikidata)

    # wikidata = wikidata['text'][int(begin):int(end)]
    

    

    tok = Tokenizer()
    vocab = Vocab()
    print("fitting the vocab")
    vocab.fit(tok.words(wikidata), baroni)


# BERT METHOD


    if save_vocab:
        with open('../data_distrembed/onetenth_vocab.pickle', 'wb') as f:
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

        torch.cuda.empty_cache()
    
    if save_vocab:
        torch.save(embavg, "../data_distrembed/onetenth_vocab.embavg.pt")
        # embavg = torch.load('../data_distrembed/first10.avgs.pt')


# EMPIRICAL METHOD


    context_dict = Context_dict()
    context_dict.fit(tok.words(wikidata), baroni)

    ft = fasttext.load_model("../data/cc.en.100.bin")

    # Calculate number of batches 
    n_batches = 1 + (len(wikidata[:]) // batch_size)
    
    for k in tqdm(range(n_batches)):
        # grab a batch_size chunk from seqs (wiki data)
        seqb = wikidata[batch_size*k:batch_size*(k+1)]
        words, _ = tok(seqb)
        all_text = [word for sentence in words for word in sentence]
        context_dict._update(all_text, baroni, window)

    print("calculate covariance")
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

    with open(f'../data_shared/fixed/df_curated{max_context}_diag_{is_diagonal}.pickle', 'wb') as handle:
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
    
    with open(f'../data_shared/df_AP{max_context}_random_{is_diagonal}.pickle', 'wb') as f:
        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)




if __name__ == '__main__':
    main()
