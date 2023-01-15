from datasets import load_dataset

# From: https://huggingface.co/datasets/wmt14
# which is a part of: https://github.com/huggingface/datasets
dataset = load_dataset("wmt14", "de-en")

# use this for bleu score: https://pytorch.org/text/stable/data_metrics.html

print(dataset)