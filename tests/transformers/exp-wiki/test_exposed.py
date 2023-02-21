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
# token length matched for convenience
txt2 = "Joe Biden was an American lawyer, politician, and a Democrat President"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokens = tokenizer(txt)['input_ids']
tokens2 = tokenizer(txt2)['input_ids']

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

# Change weight
DEL_WEIGHTS = False
if DEL_WEIGHTS:
    weight_dict = {}
    with open("model/weights.pkl", 'rb') as fhand: 
        weight_dict = pickle.load(fhand)
        indices_to_delete = [i for i in range(5, 13)]
        prefixes_to_delete = [
            'KeyWeights',
            'QueryWeights'
        ]
        for pre in prefixes_to_delete:
            for i in indices_to_delete:
                del weight_dict[pre + "$" + str(i)]

    with open("model/weights.pkl", 'wb') as fhand: 
        pickle.dump(weight_dict, fhand)

# Convert xml to onnx
agrippa.export(proj_folder, onnx_fname, bindings=bindings, log=True)

print("Search for FinalDecoderAdd:")
logs = agrippa.utils.search_log("FinalDecoderAdd")
print(logs)
print()
print("Search for AttentionSoftmax")
logs = agrippa.utils.search_log("AttentionSoftmax")
print(logs)

torch_model = agrippa.onnx_to_torch(onnx_fname)

def get_data_from_tokens(toks):
    bos_token = 50256
    chopped = torch.tensor(toks[:-1])
    to_cat = torch.full((1,), bos_token)
    x = torch.cat((to_cat, chopped), -1)
    data = F.one_hot(x, num_classes=50257).float()
    return data

data = get_data_from_tokens(tokens)
data2 = get_data_from_tokens(tokens2)

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
outputs2 = torch_model(data2, mask, scale, posembeddingmatrix, embed_scale, ln_eps)

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
print("Last layer:")
log_prob_at(1)
# print("Penultimate layer:")
# log_prob_at(2)


# For last (Republican)
token_labs = ["R-can", "person", "Democrat", "the", "eos", "chair", "good", "liar"]
tokens =     [3415,     1048,    9755,       1169,  50256, 5118,     922,    31866]
token_pos = -1
"""
# For "lawyer" (__ was an American [lawyer])
token_labs = ["lawyer", "person", "the", "eos", "chair", "politician", "novelist", "American"]
tokens =     [6853,     1048,     1169,  50256, 5118,     14971,        37986,     1605]
token_pos = 6
"""
for l in range(1, 7):
    output_num = 7 - l
    print()
    print(f"Probs at layer {l}")
    print()
    print("For Lincoln:")
    print()
    for i in range(len(tokens)):
        prob = round(torch.exp(outputs[output_num][token_pos][tokens[i]]).item() / torch.sum(torch.exp(outputs[output_num][-1])).item(), 5)
        print(f"{token_labs[i]}: {prob}")
    print()
    print("For Biden:")
    print()
    for i in range(len(tokens)):
        prob = round(torch.exp(outputs2[output_num][-2][tokens[i]]).item() / torch.sum(torch.exp(outputs2[output_num][-2])).item(), 5)
        print(f"{token_labs[i]}: {prob}")


exit(0)

for l in range(1, 9):
    output_num = 7 + l - 1
    print()
    print(f"Attention Matrix At {l}")
    print(round(outputs[output_num][-1][2].item(), 5))

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
    