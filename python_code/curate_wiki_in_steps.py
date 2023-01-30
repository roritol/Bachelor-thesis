from collections import Counter
from tqdm import trange
from tqdm import tqdm
import pickle5 as pickle 
import random
import datasets
import sys
from utility import import_baroni, text_preprocessing

def main ():
    neg_file = "../data_shared/eacl2012-data/negative-examples.txtinput"
    pos_file = "../data_shared/eacl2012-data/positive-examples.txtinput"
    results_neg_file, results_pos_file, baroni, baroni_set = import_baroni(neg_file, pos_file)

    max_length = 40
    max_context = int(sys.argv[1])
    save_file = "ma31jan"
    # begin = 50000
    # end = 2000000

    wikidata = datasets.load_dataset('wikipedia', '20200501.en')
    # wikidata = wikidata['train']['text'][int(begin):int(end)]
    wikidata = wikidata['train']['text']

    print("truncating the scentences")
    wikidata = [sentence[:max_length].strip() if len(sentence.split()) > max_length else sentence.strip()
            for seq in tqdm(wikidata)
            for sentence in seq.split(".")]

    
    # Shuffle the order of the sentences in wikidata
    print("start the loops")
    for i in range(1, 6):
        for max_context in range(25,51,5):
            collected_sentences = []
            sentence_counter = {word: int(0) for word in baroni_set}
            random.shuffle(wikidata)
            # Iterate through the shuffled list of sentences
            for sentence in tqdm(wikidata):
                words = sentence.split()
                for word in words:
                    if word in baroni_set and sentence_counter[word] < int(max_context):
                        
                            collected_sentences.append(text_preprocessing(sentence))
                            sentence_counter[word] += 1
                            break


            with open(f'../data_shared/{save_file}/curated{max_context}num{i}.pickle', 'wb') as handle:
                pickle.dump(collected_sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)

            print("collected_sentences    :" , collected_sentences[:10])
            print("sentence_counter       :", sentence_counter)
            print(f"sentence_counter length {len(sentence_counter)} baroni set length {len(baroni_set)}")
            print("max_context            :", max_context)
            print("max_length sentence    :", max_length)

if __name__ == '__main__':
    main()
