import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import pickle5 as pickle
import fasttext
import fasttext.util
from utility import cosine_similarity, create_context_dict, text_preprocessing, addDiagonal, create_combined_subset, import_baroni
from sklearn.metrics import average_precision_score
import ast

def calculate_covariance(context_dict, baroni_set, ft, window):
    covariance = {}

    for word in tqdm(baroni_set):
        total = np.zeros((100,100))

        for c_word in context_dict[word]:
            total += np.outer((ft.get_word_vector(c_word) - ft.get_word_vector(word)), (ft.get_word_vector(c_word) - ft.get_word_vector(word)))

        covariance[word] = (total / (len(context_dict[word]) * window))

    return covariance

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


# with open('context_dict1.pickle', 'wb') as handle:
#     pickle.dump(context_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('context_dict1.pickle', 'rb') as f:
#         context_dict = pickle.load(f)

def main():
    
    neg_file = "../Data_Shared/eacl2012-data/negative-examples.txtinput"
    pos_file = "../Data_Shared/eacl2012-data/positive-examples.txtinput"
    results_neg_file, results_pos_file, baroni, baroni_set = import_baroni(neg_file, pos_file)

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

