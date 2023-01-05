from collections import Counter
from tqdm import trange
from tqdm import tqdm
import pickle5 as pickle 
import random
import datasets


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

def main ():
    neg_file = "../Data_Shared/eacl2012-data/negative-examples.txtinput"
    pos_file = "../Data_Shared/eacl2012-data/positive-examples.txtinput"
    results_neg_file, results_pos_file, baroni, baroni_set = import_baroni(neg_file, pos_file)

    max_length = 40
    begin = 50000
    end = 100000

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
    for sentence in wikidata:
        
        words = sentence.split()
        for word in words:
            if word in baroni_set and sentence_counter[word] < 100:
                
                    collected_sentences.append(sentence)
                    sentence_counter[word] += 1

    with open('../data_distrembed/curated50000.pickle', 'wb') as handle:
        pickle.dump(collected_sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("collected_sentences    :" , collected_sentences[:10])
    print("sentence_counter       :", sentence_counter)
    print(f"sentence_counter length {len(sentence_counter)} baroni set length {len(baroni_set)}")

if __name__ == '__main__':
    main()
