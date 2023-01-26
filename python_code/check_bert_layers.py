# author: vlad niculae
# license simplified bsd


from typing import List
import torch
from transformers import (DistilBertTokenizerFast, DistilBertModel)


class Tokenizer:

    def __init__(self):
        self._t = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    def words(self, sequences: List[str]):
        return [s.split() for s in sequences]

    def __call__(self, sequences: List[str]):
        words = self.words(sequences)
        subw = self._t.batch_encode_plus(words,
                                         is_split_into_words=True,
                                         padding=True)
        return words, subw


device = 'cpu'
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

def main():
    tok = Tokenizer()

    model.config.output_hidden_states = True

    seqb = ["i like my cats."]
    words, subw = tok(seqb)
    mbart_input = subw.convert_to_tensors("pt").to(device=device)
    out = model(**mbart_input, return_dict=True)
    print(f"{subw=}")
    print(f"{out['last_hidden_state'].shape=}")  # n_sents x n_tokens x dim
    print(f"{out['hidden_states'][-1].shape=}")  # n_sents x n_tokens x dim

    print("first 5 numbers in the output (last hidden state) for word 'cats':")
    print(out['last_hidden_state'][0, 4, :5])

    print()
    print("same thing but using hidden_states[-1]")
    print(out['hidden_states'][-1][0, 4, :5])

    # so we can take the penultimate layer
    # print(out['hidden_states'][-2][0, 4, :5])

    # or even concatenate several layers (here, the last three)
    # concatenated_vectors = torch.cat(
            # [outputs[0, 4, :] for outputs in out['hidden_states'][-3:]])
    # print(f"{concatenated_vectors.shape=}")

    # or maybe average several layers
    # averaged_vectors = sum(
            # [outputs[0, 4, :] for outputs in out['hidden_states'][-3:]]) / 3
    # print(f"{averaged_vectors.shape=}")


if __name__ == '__main__':
    with torch.no_grad():
        main()
