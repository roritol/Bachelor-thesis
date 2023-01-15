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
# wikidata = datasets.load_dataset('wikipedia', '20200501.en')
# # make a subset
# wikidata = wikidata['train']

# print("start unpacking wiki data")
# # unpacking the wikidata
# texts = [x for x in wikidata['text']]

# wiki_all_text = []
# chunk = int(len(texts)/10)
# import time

# for text in tqdm(texts[0:(1)*chunk]):
#     # Appending preprocessed text to the "all text" list
#     text = text_preprocessing(text)
#     wiki_all_text += text

# print(f"chunk {1}")
# with open('wiki_preprocessed1.pickle', 'wb') as handle:
#     pickle.dump(wiki_all_text, handle, protocol=pickle.HIGHEST_PROTOCOL)

# del(wiki_all_text)
# wiki_all_text = []

# for text in tqdm(texts[chunk:(2)*chunk]):
#     # Appending preprocessed text to the "all text" list
#     text = text_preprocessing(text)
#     wiki_all_text += text

# print(f"chunk {2}")
# with open('wiki_preprocessed2.pickle', 'wb') as handle:
#     pickle.dump(wiki_all_text, handle, protocol=pickle.HIGHEST_PROTOCOL)

# del(wiki_all_text)
# wiki_all_text = []

# for text in tqdm(texts[(2)*chunk:(3)*chunk]):
#     # Appending preprocessed text to the "all text" list
#     text = text_preprocessing(text)
#     wiki_all_text += text


# print(f"chunk {3}")
# with open('wiki_preprocessed3.pickle', 'wb') as handle:
#     pickle.dump(wiki_all_text, handle, protocol=pickle.HIGHEST_PROTOCOL)

# del(wiki_all_text)
# wiki_all_text = []

# for text in tqdm(texts[(3)*chunk:(4)*chunk]):
#     # Appending preprocessed text to the "all text" list
#     text = text_preprocessing(text)
#     wiki_all_text += text

# print(f"chunk {4}")
# with open('wiki_preprocessed4.pickle', 'wb') as handle:
#     pickle.dump(wiki_all_text, handle, protocol=pickle.HIGHEST_PROTOCOL)

# del(wiki_all_text)
# wiki_all_text = []

# for text in tqdm(texts[(4)*chunk:(5)*chunk]):
#     # Appending preprocessed text to the "all text" list
#     text = text_preprocessing(text)
#     wiki_all_text += text

# print(f"chunk {5}")

# with open('wiki_preprocessed5.pickle', 'wb') as handle:
#     pickle.dump(wiki_all_text, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("open pickeled data:")

with open('wiki_preprocessed1.pickle', 'rb') as f:
        wiki_preprocessed1 = pickle.load(f)

wiki_count1 = Counter(tqdm(wiki_preprocessed1))
baroniwiki_count1 = Counter({k: wiki_count1.get(k, None) for k in baroni_set})
del(wiki_preprocessed1)
del(wiki_count1)

with open('wiki_preprocessed2.pickle', 'rb') as f:
        wiki_preprocessed2 = pickle.load(f)

wiki_count2 = Counter(tqdm(wiki_preprocessed2))
baroniwiki_count2 = Counter({k: wiki_count2.get(k, None) for k in baroni_set})
del(wiki_preprocessed2)
del(wiki_count2)

with open('wiki_preprocessed3.pickle', 'rb') as f:
        wiki_preprocessed3 = pickle.load(f)

wiki_count3 = Counter(tqdm(wiki_preprocessed3))
baroniwiki_count3 = Counter({k: wiki_count3.get(k, None) for k in baroni_set})
del(wiki_preprocessed3)
del(wiki_count3)

with open('wiki_preprocessed4.pickle', 'rb') as f:
        wiki_preprocessed4 = pickle.load(f)

wiki_count4 = Counter(tqdm(wiki_preprocessed4))
baroniwiki_count4 = Counter({k: wiki_count4.get(k, None) for k in baroni_set})
del(wiki_preprocessed4)
del(wiki_count4)

with open('wiki_preprocessed5.pickle', 'rb') as f:
        wiki_preprocessed5 = pickle.load(f)

wiki_count5 = Counter(tqdm(wiki_preprocessed5))
baroniwiki_count5 = Counter({k: wiki_count5.get(k, None) for k in baroni_set})
del(wiki_preprocessed5)
del(wiki_count5)

print("create a wiki counter")

baroniwiki_count = baroniwiki_count1 + baroniwiki_count2 + baroniwiki_count3 + baroniwiki_count4 + baroniwiki_count5

with open('baroniwiki_count.pickle', 'wb') as handle:
    pickle.dump(baroniwiki_count, handle, protocol=pickle.HIGHEST_PROTOCOL)


# # creating a context dictionairy usin fastext
# context_dict = create_context_dict(wiki_all_text, window=2)

# # get pre trained fastext model code is now replaced with a load file
# ft = fasttext.load_model("././Data/ft_reduced_100.bin")



