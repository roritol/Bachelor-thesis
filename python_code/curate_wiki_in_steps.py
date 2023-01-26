
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

    wikidata = datasets.load_dataset('wikipedia', '20200501.en')
    # wikidata = wikidata['train']['text'][int(begin):int(end)]
    wikidata = wikidata['train']['text']

    print("truncating the scentences")
    wikidata = [sentence[:max_length].strip() if len(sentence.split()) > max_length else sentence.strip()
            for seq in tqdm(wikidata)
            for sentence in seq.split(".")]

    collected_sentences = []
    sentence_counter = {word: int(0) for word in baroni_set}
    
    # Shuffle the order of the sentences in wikidata
    random.shuffle(wikidata)
    
    # Iterate through the shuffled list of sentences
    max_context = 0
    counter = 0 
    for sentence in tqdm(wikidata):
        if max_context == 30:
            break
        
        words = sentence.split()
        for word in words:
            if word in baroni_set and sentence_counter[word] < int(max_context):
                
                    collected_sentences.append(text_preprocessing(sentence))
                    sentence_counter[word] += 1

                    if all(val == max_context for val in sentence_counter.values()):
                        print("All keys have reached their max context.")
                        with open(f'../data_shared/fixed/ramdom_curated0-25/curated{max_context}num{counter}.pickle', 'wb') as handle:
                            pickle.dump(collected_sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)

                        collected_sentences = []
                        sentence_counter = {word: int(0) for word in baroni_set}
                        counter += 1
                        if counter == 6:
                            counter = 0
                            max_context += 5
                    
                    break
    
           
        
        
        
        
        

    print("max_context   :" , max_context)
    print("max_length sentence    :", max_length)
    print("sentence_counter       :", sentence_counter)
    print(f"sentence_counter length {len(sentence_counter)} baroni set length {len(baroni_set)}")
        

if __name__ == '__main__':
    main()