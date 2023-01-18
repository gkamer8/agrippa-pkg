from datasets import load_dataset
from transformers import GPT2Tokenizer
from preprocess import load_data, get_str_from_ids
from train import bindings
from data_constants import BATCH_SIZE, SEQ_LENGTH, bos_token
import random
import torch
import sys
sys.path.append('../../../src')  # where the agrippa source code is stored
import torch.nn.functional as F
import math
import agrippa
import numpy as np

# From: https://huggingface.co/datasets/wmt14
# which is a part of: https://github.com/huggingface/datasets

# use this for bleu score: https://pytorch.org/text/stable/data_metrics.html

# On brief exploration, looks like a good max number of tokens would be 256.

if __name__ == '__main__':
    device = "cpu"

    max_len = 0
    german = ""
    english = ""
    other_batch, english_batch = next(load_data())

    rand_row = random.randrange(len(other_batch))

    english_data = F.one_hot(english_batch.to(torch.int64)).float()

    # (Batch, Seq length)
    chopped = other_batch[:, :-1]
    to_cat = torch.full((BATCH_SIZE, 1), bos_token).to(device)
    other_data = torch.cat((to_cat, chopped), -1).to(torch.int64)
    other_data = F.one_hot(other_data.to(torch.int64)).float()

    example = (other_batch[rand_row], english_batch[rand_row])

    print("Other text:")
    print(get_str_from_ids(example[0]))
    print()
    print("English text:")
    print(get_str_from_ids(example[1]))

    print("Exporting model...")
    # agrippa.export("model", "transformer.onnx", index="transformer.agr", bindings=bindings, reinit=False, suppress=True)
    print("Exported")

    # Straight from Vaswani et al
    posembeddingmatrix = torch.empty((BATCH_SIZE, bindings['ntokens'], bindings['dmodel']))
    for pos in range(len(posembeddingmatrix[0])):
        for i in range(len(posembeddingmatrix[0][0])):
            # Even indices get one thing, odd the other
            if i % 2 == 0:
                posembeddingmatrix[:, pos, i] = math.sin(pos/(10_000**(i/bindings['dmodel'])))
            else:
                posembeddingmatrix[:, pos, i] = math.cos(pos/(10_000**(i/bindings['dmodel'])))
    posembeddingmatrix = posembeddingmatrix.to(device)

    proto_mask = torch.full((BATCH_SIZE, bindings['ntokens'], bindings['ntokens']), -float("inf"))
    proto_mask[:] = torch.triu(proto_mask[0], diagonal=1)
    mask = proto_mask
    mask = mask.to(device)

    torch_model = agrippa.onnx_to_torch("transformer.onnx")
    torch_model = torch_model.to(device)

    zeros_mask = torch.full((BATCH_SIZE, bindings['ntokens'], bindings['ntokens']), 0.).to(device)

    outputs = torch_model(other_data[rand_row], english_data[rand_row], mask[rand_row], zeros_mask[rand_row], posembeddingmatrix[rand_row])
    
    k = 20

    topk = torch.topk(outputs, k=k, dim=1)[1]  # gets indices of top 2 in tensor of shape (seq length, 2)
    tops = topk[:, 0].flatten()

    print()
    print("Top choice of translated output:")

    print(get_str_from_ids(tops))

    # Now doing sampling
    print()
    print("Sampling:")

    temperature = .1
    generation = torch.tensor([bos_token for _ in range(SEQ_LENGTH)])
    for i in range(SEQ_LENGTH):
        data = F.one_hot(generation, num_classes=50257).float()

        print(posembeddingmatrix[rand_row])
        exit(0)

        outputs = torch_model(data, english_data[rand_row], mask[rand_row], zeros_mask[rand_row], posembeddingmatrix[rand_row])

        sorted_logits, sorted_indices = torch.sort(outputs[i], descending=True)

        linears = sorted_logits[:k]

        # linears = outputs[1][i]
        linears /= temperature
        maxed = F.softmax(linears, dim=0)

        chosen = np.random.choice(sorted_indices[:k], p=maxed.detach().numpy())
        # chosen = np.random.choice(sorted_indices[:k])  # uniform among top k
        generation[i] = chosen

        if chosen == bos_token and i > 4:  # 4 is arbitrary, just let it cook uk
            break

    print(get_str_from_ids(generation))

