import re
import numpy as np  
from tqdm import tqdm
from numpy.linalg import norm
from scipy.spatial import distance

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


def create_context_dict(all_text, window = 1):

    # Creating a dictionary entry for each word in the texts
    context_dict = { i : list() for i in set(all_text)}


    for i, word in tqdm(enumerate(all_text)):
        for w in range(window):
            # Getting the context that is ahead by *window* words
            
            if i + 1 + w < len(all_text):
                context_dict[word].append(all_text[(i + 1 + w)]) 
            # Getting the context that is behind by *window* words    
            if i - w - 1 >= 0:
                context_dict[word].append(all_text[(i - w - 1)])


    return context_dict

def cosine_similarity(a, b):
    nominator = np.dot(a, b)
    
    a_norm = np.sqrt(np.sum(a**2))
    b_norm = np.sqrt(np.sum(b**2))
    
    denominator = a_norm * b_norm
    
    cosine_similarity = nominator / denominator
    
    return cosine_similarity