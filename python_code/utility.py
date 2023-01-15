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

def cosine_similarity(a, b):
    nominator = np.dot(a, b)
    
    a_norm = np.sqrt(np.sum(a**2))
    b_norm = np.sqrt(np.sum(b**2))
    
    denominator = a_norm * b_norm
    
    cosine_similarity = nominator / denominator
    
    return cosine_similarity

def calculate_kl_bert(wordpair, embavg, is_diagonal, vocab):
    # Get the mean vectors and covariance matrices for the two words in the word pair
    mean1, covariance_matrix1 = embavg.get_mean_covariance(vocab._tok_to_id.get(wordpair[0])) 
    mean2, covariance_matrix2 = embavg.get_mean_covariance(vocab._tok_to_id.get(wordpair[1])) 
    
    # Create PyTorch multivariate normal distributions using the mean vectors and covariance matrices
    if is_diagonal:
        p = torch.distributions.multivariate_normal.MultivariateNormal(mean1, covariance_matrix=torch.diagflat(torch.diag(covariance_matrix1)))
        q = torch.distributions.multivariate_normal.MultivariateNormal(mean2, covariance_matrix=torch.diagflat(torch.diag(covariance_matrix2)))
    else:
        p = torch.distributions.multivariate_normal.MultivariateNormal(mean1, covariance_matrix=covariance_matrix1)
        q = torch.distributions.multivariate_normal.MultivariateNormal(mean2, covariance_matrix=covariance_matrix2)

    # Calculate the KL divergence between the two distributions
    kl = torch.distributions.kl.kl_divergence(p, q)

    return kl.item()

def bert_cosine_similarity(a, b):
    nominator = torch.dot(a, b)
    
    a_norm = torch.sqrt(torch.sum(a**2))
    b_norm = torch.sqrt(torch.sum(b**2))
    
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


def create_combined_subset(results_neg_file, results_pos_file, vocab):
    baroni_pos_subset = [x for x in results_pos_file if x[0] in vocab._tok_counts and x[1] in vocab._tok_counts]
    baroni_neg_subset = [x for x in results_neg_file if x[0] in vocab._tok_counts and x[1] in vocab._tok_counts]

    baroni_subset_label = []
    
    for i in baroni_pos_subset:
        baroni_subset_label.append([i, 1])

    for i in baroni_neg_subset:
        baroni_subset_label.append([i, 0])

    return baroni_pos_subset, baroni_neg_subset, baroni_subset_label


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

# For the Emperical method

def calculate_covariance(context_dict, ft, window):
    covariance = {}

    for word, context in context_dict.items():
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


def calculate_kl_emp(covariance, ft, wordpair, is_diagonal):
    mean1 = torch.from_numpy(ft.get_word_vector(wordpair[0]))
    covariance_matrix1 = covariance[wordpair[0]]
    covariance_matrix1 = addDiagonal(covariance_matrix1, 0.1)
    mean2 = torch.from_numpy(ft.get_word_vector(wordpair[1]))
    covariance_matrix2 = covariance[wordpair[1]]
    covariance_matrix2 = addDiagonal(covariance_matrix2, 0.1)

    if is_diagonal:
        p = torch.distributions.multivariate_normal.MultivariateNormal(mean1, covariance_matrix=covariance_matrix1)
        q = torch.distributions.multivariate_normal.MultivariateNormal(mean2, covariance_matrix=covariance_matrix2)
    else:
        p = torch.distributions.multivariate_normal.MultivariateNormal(mean1, covariance_matrix=torch.diagflat(torch.diag(covariance_matrix1)))
        q = torch.distributions.multivariate_normal.MultivariateNormal(mean2, covariance_matrix=torch.diagflat(torch.diag(covariance_matrix2)))

    return float(torch.distributions.kl.kl_divergence(p, q))