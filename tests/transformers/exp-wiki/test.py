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

txt = """This eBook is for the use of anyone anywhere in the United States and
most other parts of the world at no cost and with almost no restrictions
whatsoever. You may copy it, give it away or re-use it"""

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokens = tokenizer(txt)['input_ids']

seq_length = len(tokens)

# Base model from vaswani et al
bindings = {
    'ntokens': seq_length,
    'nvocab': 50257,
    'dmodel': 512,
    'dffnhidden': 2048,
    'dvalues': 64,
    'dqueries': 64,
    'dkeys': 64,
    'nheads': 8,
    'nlayers': 6
}

# Convert xml to onnx
agrippa.export(proj_folder, onnx_fname, bindings=bindings, reinit=False, suppress=True)

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

print(outputs[0])

topk = torch.topk(outputs[0], k=20, dim=1)[1]

tops = topk[:, 0].flatten()

y = tokenizer.convert_ids_to_tokens(list(tops))

print("Target:")
print(txt)

my_str = tokenizer.convert_tokens_to_string(y)
print("First STR:")
print(my_str)

tops = topk[:, 1].flatten()

y = tokenizer.convert_ids_to_tokens(list(tops))

my_str = tokenizer.convert_tokens_to_string(y)
print("Second STR:")
print(my_str)

print("SAMPLING with prompt:")
prompt = "Despite various publications of results where hand-washing reduced mortality to below 1%, Semmelweis's observations conflicted with the established"
print(prompt)
print()
prompt_ids = tokenizer(prompt)['input_ids']

extended_data = torch.tensor(prompt_ids + [bos_token for _ in range(seq_length-len(prompt_ids))])
ext_one_hot = F.one_hot(extended_data, num_classes=50257).float()
outputs = torch_model(ext_one_hot, mask, scale, posembeddingmatrix, embed_scale, ln_eps)
topk = torch.topk(outputs[0], k=20, dim=1)[1]
tops = topk[:, 0].flatten()
y = tokenizer.convert_ids_to_tokens(list(tops))
my_str = tokenizer.convert_tokens_to_string(y)
print("Tops before sampling:")
print(my_str)
print()
print("Last token:")
print(tokenizer.convert_tokens_to_string([y[-1]]))
print()

temperature = 1
generation = torch.tensor(prompt_ids + [bos_token for _ in range(seq_length-len(prompt_ids))])
token_list = [i for i in range(bindings['nvocab'])]
for i in range(len(prompt_ids), seq_length):
    data = F.one_hot(generation, num_classes=50257).float()
    outputs = torch_model(data, mask, scale, posembeddingmatrix, embed_scale, ln_eps)
    linears = outputs[1][i]
    linears /= temperature
    maxed = F.softmax(linears, dim=0)
    chosen = np.random.choice(token_list, p=maxed.detach().numpy())
    generation[i] = chosen
    chosen_token = tokenizer.convert_ids_to_tokens([chosen])
    my_str = tokenizer.convert_tokens_to_string(chosen_token)

all_tokens = tokenizer.convert_ids_to_tokens(generation)
my_str = tokenizer.convert_tokens_to_string(all_tokens)
print(my_str)

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
    