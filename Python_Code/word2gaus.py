import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import pickle
import tensorflow as tf
import fasttext
import fasttext.util
from utility import text_preprocessing, create_context_dict
import datasets
from collections import Counter

# import baroni
neg_file = "../Data_Shared/eacl2012-data/negative-examples.txtinput"
pos_file = "../Data_Shared/eacl2012-data/positive-examples.txtinput"
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

# set wikidata to the subset of wikipedia dataset
wikidata = datasets.load_dataset('wikipedia', '20200501.en')
# make a subset
wikidata = wikidata['train']

print("start unpacking wiki data")
# unpacking the wikidata
texts = [x for x in wikidata['text']]

wiki_all_text = []

for text in tqdm(texts):
    # Appending preprocessed text to the "all text" list
    text = text_preprocessing(text)
    wiki_all_text += text

print("create a wiki counter")
wiki_count = Counter(tqdm(wiki_all_text))

baroniwiki_count = {k: wiki_count.get(k, None) for k in baroni_set}

with open('baroniwiki_count.pickle', 'wb') as handle:
    pickle.dump(baroniwiki_count, handle, protocol=pickle.HIGHEST_PROTOCOL)


# # creating a context dictionairy usin fastext
# context_dict = create_context_dict(wiki_all_text, window=2)

# # get pre trained fastext model code is now replaced with a load file
# ft = fasttext.load_model("././Data/ft_reduced_100.bin")



