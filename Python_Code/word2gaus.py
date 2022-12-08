import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import pickle
import tensorflow as tf
import fasttext
import fasttext.util
from utility import text_preprocessing, create_context_dict, cosine_similarity
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

# get pre trained fastext model code is now replaced with a load file
# fasttext.load_model('cc.en.300.bin')
ft = fasttext.load_model("././Data/ft_reduced_100.bin")

print("open pickeled data:")
with open('wiki_preprocessed1.pickle', 'rb') as f:
        wiki_preprocessed1 = pickle.load(f)

print(("amount of words: ", len(wiki_preprocessed1)))

# creating a context dictionairy using fastext
print("start context dict")
window = 5
context_dict = create_context_dict(wiki_preprocessed1, window)

with open('context_dict1.pickle', 'wb') as handle:
    pickle.dump(context_dict1, handle, protocol=pickle.HIGHEST_PROTOCOL)

total = np.zeros((300,300))

covariance = {}

for word in tqdm(baroni_set):
    for c_word in context_dict[word]:
        w = ft.get_word_vector(word)
        total += np.outer((ft.get_word_vector(c_word) - w), (ft.get_word_vector(c_word) - w))
    
    covariance[word] = (total / (len(context_dict[word]) * window))

baroni_subset_label = []

for i in results_pos_file:
    baroni_subset_label.append([i, 1])

for i in results_neg_file:
    baroni_subset_label.append([i, 0])

df1 = pd.DataFrame(baroni_subset_label, columns =['Wordpair', 'True label'])


baroni_subset_kl = []

for wordpair in (results_pos_file + results_neg_file):
    mean1 = torch.from_numpy(ft.get_word_vector(wordpair[0]))
    covariance_matrix1 = torch.from_numpy(covariance[wordpair[0]])
    mean2 = torch.from_numpy(ft.get_word_vector(wordpair[1]))
    covariance_matrix2 = torch.from_numpy(covariance[wordpair[1]])
    
    p = torch.distributions.multivariate_normal.MultivariateNormal(mean1, covariance_matrix=covariance_matrix1)
    q = torch.distributions.multivariate_normal.MultivariateNormal(mean2, covariance_matrix=covariance_matrix2)

    baroni_subset_kl.append(float(torch.distributions.kl.kl_divergence(p, q)))

df1['KL score'] = baroni_subset_kl


baroni_subset_cos = []

for wordpair in (baroni_pos_subset + baroni_neg_subset):
    A = ft.get_word_vector(wordpair[0])
    B = ft.get_word_vector(wordpair[1])
    baroni_subset_cos.append(cosine_similarity(A, B))

df1['COS score'] = baroni_subset_cos

# with open('wiki_preprocessed2.pickle', 'rb') as f:
#         wiki_preprocessed2 = pickle.load(f)


# with open('wiki_preprocessed3.pickle', 'rb') as f:
#         wiki_preprocessed3 = pickle.load(f)


# with open('wiki_preprocessed4.pickle', 'rb') as f:
#         wiki_preprocessed4 = pickle.load(f)


# with open('wiki_preprocessed5.pickle', 'rb') as f:
#         wiki_preprocessed5 = pickle.load(f)


with open('df1.pickle', 'wb') as handle:
    pickle.dump(df1, handle, protocol=pickle.HIGHEST_PROTOCOL)



