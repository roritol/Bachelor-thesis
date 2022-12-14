import re
import numpy as np  
from tqdm import tqdm
from numpy.linalg import norm
from scipy.spatial import distance
import torch
from collections import Counter

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

def text_preprocessing(
    text:list,
    punctuations = r'''!()-[]{};:'"\,<>./?@#$%^&*_“~''',
    stop_words=['and', 'a', 'is', 'the', 'in', 'be', 'will']
    )->list:
    """
    A method to preproces text
    """
    for x in text.lower(): 
        if x in punctuations: 
            text = text.replace(x, "")

    # Removing words that have numbers in them
    text = re.sub(r'\w*\d\w*', '', text)

    # Removing digits
    text = re.sub(r'[0-9]+', '', text)

    # Cleaning the whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Setting every word to lower
    text = text.lower()

    # Converting all our text to a list 
    text = text.split(' ')

    # Droping empty strings
    text = [x for x in text if x!='']

    # Droping stop words
#     text = [x for x in text if x not in stop_words]

    return text

# def create_context_dict(all_text, window = 1):

#     # Creating a dictionary entry for each word in the texts
#     context_dict = { i : list() for i in set(all_text)}


#     for i, word in tqdm(enumerate(all_text)):
#         for w in range(window):
#             # Getting the context that is ahead by *window* words
#             if i + 1 + w < len(all_text):
#                 context_dict[word].append(all_text[(i + 1 + w)]) 
#             # Getting the context that is behind by *window* words    
#             if i - w - 1 >= 0:
#                 context_dict[word].append(all_text[(i - w - 1)])

#     return context_dict


def cosine_similarity(a, b):
    nominator = np.dot(a, b)
    
    a_norm = np.sqrt(np.sum(a**2))
    b_norm = np.sqrt(np.sum(b**2))
    
    denominator = a_norm * b_norm
    
    cosine_similarity = nominator / denominator
    
    return cosine_similarity

#  In practice, it is also necessary to add a small ridge term 
#  δ > 0 to the diagonal of the matrix to regularize and avoid 
#  numerical problems when inverting - vilnis mccalumn

def addDiagonal(matrix, x):
    assert x < 1, f"x greater than 0 expected, got: {x}"
    
    for i in range(len(matrix)):
        matrix[i][i] = matrix[i][i] + x
    
    return matrix


def create_combined_subset(context_dict, results_neg_file, results_pos_file, combined_set):
    combined_set_context_len = {}
    
    for i in combined_set:
        combined_set_context_len[i] = len(context_dict[i])

    combined_set_30plus = [x for x , key in combined_set_context_len.items() if key > 30]
    
    baroni_pos_subset = [x for x in results_pos_file if x[0] in combined_set_30plus and x[1] in combined_set_30plus]
    baroni_neg_subset = [x for x in results_neg_file if x[0] in combined_set_30plus and x[1] in combined_set_30plus]

    return baroni_pos_subset, baroni_neg_subset


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


    