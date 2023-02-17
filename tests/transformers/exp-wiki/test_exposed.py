import sys
sys.path.append('../../../src')  # where the agrippa source code is stored
import agrippa
import onnxruntime as ort
from transformers import GPT2Tokenizer
import torch.nn.functional as F
import torch
import math
import pickle
import numpy as np
import os

proj_folder = "model"
onnx_fname = "test-decoder.onnx"

#txt = """The die is cast; I have consented to return if we are not destroyed.
#Thus are my hopes blasted by cowardice and"""

txt = """Abraham Lincoln was an American lawyer, politician, and a Republican"""

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokens = tokenizer(txt)['input_ids']

seq_length = len(tokens)

# Base model from vaswani et al
bindings = {
    'ntokens': seq_length,
    'nvocab': 50257,
    'dmodel': 1024,
    'dffnhidden': 4096,
    'dvalues': 64,
    'dqueries': 64,
    'dkeys': 64,
    'nheads': 16,
    'nlayers': 6
}

# Convert xml to onnx
agrippa.export(proj_folder, onnx_fname, bindings=bindings, log=True)

logs = agrippa.utils.search_log("FinalDecoderAdd")

print(logs)

torch_model = agrippa.onnx_to_torch(onnx_fname)

bos_token = 50256
chopped = torch.tensor(tokens[:-1])
to_cat = torch.full((1,), bos_token)
x = torch.cat((to_cat, chopped), -1)
data = F.one_hot(x, num_classes=50257).float()

scale = math.sqrt(bindings['dmodel'])
embed_scale = math.sqrt(bindings['dmodel'])
ln_eps = 1e-5

proto_mask = torch.full((bindings['ntokens'], bindings['ntokens']), -float("inf"))
proto_mask = torch.triu(proto_mask, diagonal=1)
mask = proto_mask

# Straight from Vaswani et al
posembeddingmatrix = torch.empty((bindings['ntokens'], bindings['dmodel']))
for pos in range(len(posembeddingmatrix)):
    for i in range(len(posembeddingmatrix[0])):
        # Even indices get one thing, odd the other
        if i % 2 == 0:
            posembeddingmatrix[pos, i] = math.sin(pos/(10_000**(i/bindings['dmodel'])))
        else:
            posembeddingmatrix[pos, i] = math.cos(pos/(10_000**(i/bindings['dmodel'])))

# Make predictions for this batch
outputs = torch_model(data, mask, scale, posembeddingmatrix, embed_scale, ln_eps)

def log_prob_at(layer):

    topk = torch.topk(outputs[layer], k=20, dim=1)[1]

    tops = topk[:, 0].flatten()

    y = tokenizer.convert_ids_to_tokens(list(tops))

    print("Target:")
    print(txt)

    print("(target) | (prediction1)")
    print()
    for i in range(len(outputs[layer])):
        tar = tokenizer.convert_ids_to_tokens([tokens[i]])
        tar = tokenizer.convert_tokens_to_string(tar)
        pred = tops[i]
        pred = tokenizer.convert_ids_to_tokens([pred])
        pred = tokenizer.convert_tokens_to_string(pred)
        print(f"({tar}) | ({pred})")

    print()
    print("(target) | (prediction2)")
    print()

    tops = topk[:, 1].flatten()
    for i in range(len(outputs[layer])):
        tar = tokenizer.convert_ids_to_tokens([tokens[i]])
        tar = tokenizer.convert_tokens_to_string(tar)
        pred = tops[i]
        pred = tokenizer.convert_ids_to_tokens([pred])
        pred = tokenizer.convert_tokens_to_string(pred)
        print(f"({tar}) | ({pred})")

# 25777 is Republican
# print("Last layer:")
# log_prob_at(1)
# print("Penultimate layer:")
# log_prob_at(2)

token_labs = ["R-can", "person", "Democrat", "the", "eos", "chair", "good", "liar"]
tokens =     [3415,     1048,    9755,       1169,  50256, 5118,     922,    31866]
for l in range(1, 7):
    output_num = 7-l
    print()
    print(f"Log probs at layer {l}")

    for i in range(len(tokens)):
        log_prob = round(outputs[output_num][-1][tokens[i]].item(), 2)
        print(f"{token_labs[i]}: {log_prob}")

exit(0)

print()

print("SAMPLING with prompt:")
prompt = "Abraham Lincoln was an American lawyer, politician, and"
print(prompt)
print()
prompt_ids = tokenizer(prompt)['input_ids']


# Top k sampling!
k = 5

presence_penalty = 2

print("Sampling:")

temperature = 1.5
generation = torch.tensor(prompt_ids + [bos_token for _ in range(seq_length-len(prompt_ids))])
for i in range(len(prompt_ids)-1, seq_length-1):
    data = F.one_hot(generation, num_classes=50257).float()
    outputs = torch_model(data, mask, scale, posembeddingmatrix, embed_scale, ln_eps)

    # Apply presence penalty
    for id in generation:
        outputs[1][i][id] /= presence_penalty

    sorted_logits, sorted_indices = torch.sort(outputs[1][i], descending=True)

    linears = sorted_logits[:k]

    # linears = outputs[1][i]
    linears /= temperature
    maxed = F.softmax(linears, dim=0)

    chosen = np.random.choice(sorted_indices[:k], p=maxed.detach().numpy())
    # chosen = np.random.choice(sorted_indices[:k])  # uniform among top k
    generation[i+1] = chosen

all_tokens = tokenizer.convert_ids_to_tokens(generation)
my_str = tokenizer.convert_tokens_to_string(all_tokens)
print(my_str)

exit(0)

# Get a few random samples
print()
print("Arbitrary data examples...")

data_path = os.path.join("dataset", "batches")
batches = os.listdir(data_path)
with open(os.path.join(data_path, batches[0]), 'rb') as fhand:
    x = pickle.load(fhand)
    ex = x[0]
    tokens = tokenizer.convert_ids_to_tokens(ex)
    my_str = tokenizer.convert_tokens_to_string(tokens)
    print(my_str)

print()
print("Second example...")
with open(os.path.join(data_path, batches[1]), 'rb') as fhand:
    x = pickle.load(fhand)
    ex = x[0]
    tokens = tokenizer.convert_ids_to_tokens(ex)
    my_str = tokenizer.convert_tokens_to_string(tokens)
    print(my_str)

    print()
    print("Third example...")

    ex2 = x[1]
    tokens = tokenizer.convert_ids_to_tokens(ex2)
    my_str = tokenizer.convert_tokens_to_string(tokens)
    print(my_str)
    