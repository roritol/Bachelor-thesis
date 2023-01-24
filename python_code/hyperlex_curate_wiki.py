from collections import Counter
from tqdm import trange
from tqdm import tqdm
import pickle5 as pickle 
import random
import datasets
import sys
from utility import import_hyperlex
import pandas as pd

def main ():
    
    file = "../data_shared/hyperlex-data/hyperlex-all.txt"
    HyperLex = pd.DataFrame(import_hyperlex(file))

    HyperLex.columns = HyperLex.iloc[0]
    HyperLex = HyperLex.iloc[1:].reset_index(drop=True)

    HyperLex_set = set(HyperLex["WORD1"].values.tolist() + HyperLex["WORD2"].values.tolist())

    max_length = 40
    max_context = int(sys.argv[1])
    # begin = 50000
    # end = 100000

    wikidata = datasets.load_dataset('wikipedia', '20200501.en')
    # wikidata = wikidata['train']['text'][int(begin):int(end)]
    wikidata = wikidata['train']['text']

    print("truncating the scentences")
    wikidata = [sentence[:max_length].strip() if len(sentence.split()) > max_length else sentence.strip()
            for seq in tqdm(wikidata)
            for sentence in seq.split(".")]

    collected_sentences = []
    sentence_counter = {word: int(0) for word in HyperLex_set}
    # Shuffle the order of the sentences in wikidata
    random.shuffle(wikidata)
    # Iterate through the shuffled list of sentences
    for sentence in wikidata:
        
        words = sentence.split()
        for word in words:
            if word in HyperLex_set and sentence_counter[word] < int(max_context):
                
                    collected_sentences.append(sentence)
                    sentence_counter[word] += 1
                    continue
                    

    with open(f'../data_shared/hyperlex_output/curated/hyp_curated{max_context}.pickle', 'wb') as handle:
        pickle.dump(collected_sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("collected_sentences    :" , collected_sentences[:10])
    print("sentence_counter       :", sentence_counter)
    print(f"sentence_counter length {len(sentence_counter)} baroni set length {len(HyperLex_set)}")
    print("max_context            :", max_context)
    print("max_length sentence    :", max_length)

if __name__ == '__main__':
    main()
