from datasets import load_dataset
from transformers import GPT2Tokenizer
from data_constants import BATCH_SIZE, SEQ_LENGTH, device
import torch
import random
import torch.nn.functional as F

# From: https://huggingface.co/datasets/wmt14
# which is a part of: https://github.com/huggingface/datasets

# use this for bleu score: https://pytorch.org/text/stable/data_metrics.html

# On brief exploration, looks like a good max number of tokens would be 256.

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
other_lang = 'de'  # or "de"

const_zeros = torch.zeros((BATCH_SIZE, SEQ_LENGTH)).to(device)

# Some experimentation shows that the proportion of examples that are > seq length @ seq legnth = 100 is .058
#   (at 64, closer to .2)

def tokenize_and_pad(example):
    # Tokenize
    example['translation']['en'] = torch.tensor(tokenizer(example['translation']['en'])['input_ids']).to(device)
    example['translation'][other_lang] = torch.tensor(tokenizer(example['translation'][other_lang])['input_ids']).to(device)

    # Pad or remove
    def pad_or_remove(lang):
        if len(example['translation'][lang]) >= SEQ_LENGTH:
            example['translation'][lang] = example['translation'][lang][:SEQ_LENGTH]
        else:
            # src for pad: https://stackoverflow.com/questions/48686945/reshaping-a-tensor-with-padding-in-pytorch
            x = example['translation'][lang]
            example['translation'][lang] = F.pad(x, (0, SEQ_LENGTH - len(x)), 'constant', 50256).to(device)
    pad_or_remove('en')
    pad_or_remove(other_lang)
    return example

# Yields (other language batch, english batch) each of shape (BATCH_SIZE, SEQ_LENGTH)
def load_data(split="train"):

    dataset = load_dataset("wmt14", f"{other_lang}-en").shuffle()

    batch_en = torch.empty((BATCH_SIZE, SEQ_LENGTH)).to(device)
    batch_other = torch.empty((BATCH_SIZE, SEQ_LENGTH)).to(device)
    for i, example in enumerate(dataset[split]):

        x = tokenize_and_pad(example)
        batch_en[i % BATCH_SIZE] = x['translation']['en']
        batch_other[i % BATCH_SIZE] = x['translation'][other_lang]

        if i % BATCH_SIZE == BATCH_SIZE - 1:
            yield batch_other, batch_en

    print(f"Done")


def get_str_from_ids(row):
    toks = tokenizer.convert_ids_to_tokens(row)
    my_str = tokenizer.convert_tokens_to_string(toks)
    return my_str

if __name__ == '__main__':
    # For testing
    sample_size = 2
    for i, (x, y) in enumerate(load_data()):
        row = random.randrange(BATCH_SIZE)
        other = x[row]
        en = y[row]
        print()
        print("Other language:")
        print(get_str_from_ids(other))
        print()
        print("English: ")
        print(get_str_from_ids(en))

        if i + 1 >= sample_size:
            break
