from datasets import load_dataset
from transformers import GPT2Tokenizer

# From: https://huggingface.co/datasets/wmt14
# which is a part of: https://github.com/huggingface/datasets
dataset = load_dataset("wmt14", "de-en")

# use this for bleu score: https://pytorch.org/text/stable/data_metrics.html

# On brief exploration, looks like a good max number of tokens would be 256.

max_len = 0
for i, example in enumerate(dataset['train']):
    
    german = example['translation']['de']
    english = example['translation']['en']

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    en_tokens = tokenizer(english)['input_ids']
    de_tokens = tokenizer(german)['input_ids']

    max_len = max([len(en_tokens), len(de_tokens), max_len])
    if i % 50 == 0:
        print(i)
    if i > 200:
        break

print(f"Max: {max_len}")
