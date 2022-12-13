import torch
from transformers import (DistilBertTokenizerFast, DistilBertModel)

class EmbedAverages(torch.nn.Module):

    def __init__(self, n_words, dim):

        super().__init__()
        self.register_buffer("_sum", torch.zeros(n_words, dim))
        self.register_buffer("_ssq", torch.zeros(n_words, dim))
        self.register_buffer("_sum_normed", torch.zeros(n_words, dim))
        self.register_buffer("_counts", torch.zeros(n_words, dtype=torch.long))

    def add(self, ix, vec):
        # could use B.index_add(0, ix, torch.ones_like(ix, dtype=torch.float)
        self._counts[ix] += 1
        self._sum[ix] += vec
        self._ssq[ix] += vec ** 2
        self._sum_normed[ix] += vec / torch.norm(vec, dim=-1, keepdim=True)


# Load the "embavg" file saved at the end of the script
embavg = torch.load('/Users/rori/Documents/UVA_4/data_distrembed/roen.avgs.pt')


with open("/Users/rori/Documents/UVA_4/data_distrembed/roen.vocab", "r") as f:
    n_words = f.readlines()


print(len(n_words))
print(len(embavg._counts))


# # Create a dictionary that maps words to their vector representations
# word_vectors = {}

# # Iterate over the words in the vocabulary
# for i, word in enumerate(vocab._toks):
#     # Get the average vector for the current word
#     vec = embavg._sum[i]

#     # Add an entry to the dictionary that maps the current word to its average vector
#     word_vectors[word] = vec

# # Get the vector for the word "church" from the dictionary
# hello_vector = word_vectors["church"]
# print(hello_vector)