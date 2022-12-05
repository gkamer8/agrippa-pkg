import sys
sys.path.append('../../../src')  # where the agrippa source code is stored
import agrippa
import onnxruntime as ort
from transformers import GPT2Tokenizer
import torch.nn.functional as F
import torch
import math
import pickle

proj_folder = "model"
onnx_fname = "decoder.onnx"

txt = """The die is cast; I have consented to return if we are not destroyed.
Thus are my hopes blasted by cowardice and indecision; I come back
ignorant and disappointed. It requires more philosophy than I possess
to bear this injustice with patience.

September 12th.


It is past; I am returning to England. I have lost my hopes of utility
and glory; I have lost my friend. But I will endeavour to detail these
bitter circumstances to you, my dear sister; and while I am wafted
towards England and towards you, I will not despond."""

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
outputs = torch_model(data, mask, scale, posembeddingmatrix, embed_scale)

print(outputs[0])

topk = torch.topk(outputs[0], k=20, dim=1)[1]

tops = topk[:, 0].flatten()

y = tokenizer.convert_ids_to_tokens(list(tops))

my_str = tokenizer.convert_tokens_to_string(y)
print("First STR:")
print(my_str)

tops = topk[:, 19].flatten()

y = tokenizer.convert_ids_to_tokens(list(tops))

my_str = tokenizer.convert_tokens_to_string(y)
print("Second STR:")
print(my_str)