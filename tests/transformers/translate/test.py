from datasets import load_dataset
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# From: https://huggingface.co/datasets/wmt14
# which is a part of: https://github.com/huggingface/datasets
dataset = load_dataset("wmt14", "de-en").shuffle()

# use this for bleu score: https://pytorch.org/text/stable/data_metrics.html

# On brief exploration, looks like a good max number of tokens would be 256.

max_len = 0
german = ""
english = ""
for i, example in enumerate(dataset['train']):

    german += example['translation']['de'] + "<|endoftext|>"
    english += example['translation']['en'] + "<|endoftext|>"

    print(english)

    if i % 100 == 0:
        en_tokens = tokenizer(english)['input_ids']
        de_tokens = tokenizer(german)['input_ids']

    if i > 1:
        break

print(f"Done")
