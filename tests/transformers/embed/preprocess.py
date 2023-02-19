from lib2to3.pgen2 import token
from datasets import load_dataset
from transformers import GPT2Tokenizer
from data_constants import BATCH_SIZE, SEQ_LENGTH, device
import torch
import random
import torch.nn.functional as F

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

const_zeros = torch.zeros((BATCH_SIZE, SEQ_LENGTH)).to(device)


# Yields batches of [list of input_ids] (tokens) - batches are in first dimension
def load_data(split="train", batch_size=BATCH_SIZE):

    dataset = load_dataset("wikitext", "wikitext-103-v1")

    batch = torch.empty((batch_size, SEQ_LENGTH)).to(device)
    batch_i = 0  # where we are in adding the thing to the batch
    accum = []  # current sequence to be added
    for example in dataset[split]:

        txt = example['text']
        x = tokenizer(txt)['input_ids']
        accum.extend(x)

        if len(accum) >= SEQ_LENGTH:
            batch[batch_i] = torch.tensor(accum[:SEQ_LENGTH])
            accum = accum[SEQ_LENGTH:]
            batch_i += 1

        if batch_i >= batch_size:
            yield batch
            batch = torch.empty((batch_size, SEQ_LENGTH)).to(device)
            batch_i = 0

    print(f"Done with all data")


def get_str_from_ids(row):
    toks = tokenizer.convert_ids_to_tokens(row)
    my_str = tokenizer.convert_tokens_to_string(toks)
    return my_str

if __name__ == '__main__':
    # For testing
    sample_size = 2
    for i, x in enumerate(load_data()):
        print(get_str_from_ids(x[0]))
        print(get_str_from_ids(x[1]))
        exit(0)
