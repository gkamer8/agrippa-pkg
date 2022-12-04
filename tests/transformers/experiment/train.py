import pickle
import math
import sys
sys.path.append('../../../src')  # where the agrippa source code is stored

import agrippa
import onnxruntime as ort
import numpy as np
import os
import random
import math
import torch
import matplotlib.pyplot as plt
import data as my_data
import torch.nn.functional as F

batch_size = 16
seq_length = 128

my_data.chunkify("dataset", "chunks", seq_length)
my_data.batchify('dataset', 'batches', batch_size, 'chunks')

proj_folder = "model"
onnx_fname = "decoder.onnx"

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

"""
bindings = {
    'ntokens': 3,
    'nvocab': 128,
    'dmodel': 24,
    'dffnhidden': 32,
    'dvalues': 8,
    'dqueries': 8,
    'dkeys': 8,
    'nheads': 2,
    'nlayers': 2
}
"""

# Convert xml to onnx
agrippa.export(proj_folder, onnx_fname, bindings=bindings, reinit=False, suppress=True)

print("Exported")

# Now we train
torch_model = agrippa.onnx_to_torch(onnx_fname)


# Here is the idea, bruv:
# index 0 = a
# index 1 = b
# index 2 = c

# Every sequence starts with a or b (50% prob)
# If the sequence starts with an a, second is b
# Otherwise, second is c
# BOS tag is index 127

nbatches = my_data.get_n_batches(os.path.join('dataset', 'batches', 'meta.json'))
batch_order = [i for i in range(nbatches)]
np.random.shuffle(batch_order)
bos_token = 50256
def gen_data_tokens_pair():
    for i in range(len(batch_order)):
        with open(os.path.join('dataset', 'batches', f'{i}.pkl'), 'rb') as fhand:
            prospect = pickle.load(fhand)
        if len(prospect) != batch_size:
            continue
        # Just forget about sequences not of common length for now
        skip = False
        for k in range(batch_size):
            if len(prospect[k]) != bindings['ntokens']:
                skip = True
                break
        if skip:
            continue
        y = torch.tensor(prospect)
        chopped = y[:, :-1]
        to_cat = torch.full((batch_size, 1), bos_token)
        x = torch.cat((to_cat, chopped), -1)
        yield x, y


scale = math.sqrt(bindings['dkeys'])

# FIX THIS - TODO
proto_mask = torch.full((batch_size, bindings['ntokens'], bindings['ntokens']), -float("inf"))
proto_mask[:] = torch.triu(proto_mask[0], diagonal=1)
mask = proto_mask

# Straight from Vaswani et al
posembeddingmatrix = torch.empty((batch_size, bindings['ntokens'], bindings['dmodel']))
for pos in range(len(posembeddingmatrix[0])):
    for i in range(len(posembeddingmatrix[0][0])):
        # Even indices get one thing, odd the other
        if i % 2 == 0:
            posembeddingmatrix[:, pos, i] = math.sin(pos/(10_000**(i/bindings['dmodel'])))
        else:
            posembeddingmatrix[:, pos, i] = math.cos(pos/(10_000**(i/bindings['dmodel'])))


# Vaswani et al use label smoothing = 0.1
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
# loss_fn = torch.nn.MSELoss()

log_freq = 1
loss_log = []

lr_schedule = {
    0:    0.001
}

current_lr = lr_schedule[0]
optimizer = torch.optim.Adam(torch_model.parameters(), lr=current_lr, betas=(0.9, 0.98), eps=1e-09)
# optimizer = torch.optim.SGD(torch_model.parameters(), lr=current_lr)

accum_loss = 0

max_train_size = 1000

gradient_accum_steps = 2

warmup_steps = 4000

for i, (data, label) in enumerate(gen_data_tokens_pair()):

    if i % log_freq == 0:
        print(f"At step {i}")

    if i in lr_schedule:
        new_lr = bindings['dmodel'] ** (-.5) * min((i+1)**(-0.5), (i+1)*warmup_steps**(-1.5))
        optimizer = torch.optim.Adam(torch_model.parameters(), lr=new_lr, betas=(0.9, 0.98), eps=1e-09)

    # Add dimension to data
    data = F.one_hot(data).float()

    """ids_to_toks = my_data.get_ids_to_toks()
    toks = ids_to_toks(label[0])
    str = my_data.get_toks_to_str()(toks)
    print(str)
    exit(0)"""

    # Make predictions for this batch
    outputs = torch_model(data, mask, scale, posembeddingmatrix)

    """oneoutputs = torch_model(data[0], mask[0], scale, posembeddingmatrix[0])
    with open("bruh.pkl", 'wb') as fp:
        pickle.dump(data[0], fp)
    print(oneoutputs[0])
    exit(0)"""

    # Loss function expects labels in the form (Batch size, # Classes, other dimensions...)
    train_output = outputs[1].permute((0, 2, 1))

    # Compute the loss and its gradients
    loss = loss_fn(train_output, label) / gradient_accum_steps
    loss.backward()
    accum_loss += loss.item()

    if (i+1) % gradient_accum_steps == 0:
        print("Optimizing...")
        optimizer.step()
        optimizer.zero_grad()
        # Scale the loss by the gradient accumulation
        loss_log.append(accum_loss)
        accum_loss = 0

    if i >= max_train_size:
        break

def save_model():
    weights_dict = {}
    whole_dict = torch_model.state_dict()
    for key in whole_dict:
        real_name = key[len('initializers.'):]
        weights_dict[real_name] = whole_dict[key]

    with open(os.path.join("model", 'weights.pkl'), 'wb') as fhand:
        pickle.dump(weights_dict, fhand)
    

save_model()
print(loss_log[-1])
print(loss_log)

plt.plot(loss_log)
plt.show()

