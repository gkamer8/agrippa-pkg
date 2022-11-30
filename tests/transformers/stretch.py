import math
import sys
sys.path.append('../../src')  # where the agrippa source code is stored

import agrippa
import onnxruntime as ort
import numpy as np
import os
import random
import math
import torch
import matplotlib.pyplot as plt


proj_folder = "mha-stretch-proj"
onnx_fname = "decoder.onnx"

bindings = {
    'ntokens': 3,
    'nvocab': 128,
    'dmodel': 24,
    'dffnhidden': 32,
    'dvalues': 8,
    'dqueries': 8,
    'dkeys': 8,
    'nheads': 2
}

# Convert xml to onnx
agrippa.export(proj_folder, onnx_fname, bindings=bindings, reinit=True, suppress=True)

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

batch_size = 100

def gen_data_tokens_pair(new_batch_size=None):

    if new_batch_size is None:
        new_batch_size = batch_size

    label = torch.zeros((new_batch_size, bindings['ntokens'])).long()
    data = torch.zeros((new_batch_size, bindings['ntokens'], bindings['nvocab']))

    a_first = new_batch_size // 2

    label[:a_first, 0] = 0  # a
    data[:a_first, 0, 127] = 1. # <BOS>

    label[:a_first, 1] = 1  # b
    data[:a_first, 1, 0] = 1.   # a

    label[:a_first, 2] = 2  # c
    data[:a_first, 2, 1] = 1.   # b

    label[a_first:, 0] = 3  # d
    data[a_first:, 0, 127] = 1. # <BOS>

    label[a_first:, 1] = 4  # e
    data[a_first:, 1, 3] = 1.   # d

    label[a_first:, 2] = 5  # f
    data[a_first:, 2, 4] = 1.   # e
    
    return data, label


scale = math.sqrt(bindings['dmodel'])

mask = torch.empty((batch_size, bindings['ntokens'], bindings['ntokens']))
mask[:] = torch.tensor([
    [0, -float("inf"), -float("inf")],
    [0, 0, -float("inf")],
    [0, 0, 0]
])

# Straight from Vaswani et al
posembeddingmatrix = torch.empty((batch_size, bindings['ntokens'], bindings['dmodel']))
for pos in range(len(posembeddingmatrix[0])):
    for i in range(len(posembeddingmatrix[0][0])):
        # Even indices get one thing, odd the other
        if i % 2 == 0:
            posembeddingmatrix[:, pos, i] = math.sin(pos/(10_000**(i/bindings['dmodel'])))
            # posembeddingmatrix[pos][i] = 0.
        else:
            posembeddingmatrix[:, pos, i] = math.cos(pos/(10_000**(i/bindings['dmodel'])))
            # posembeddingmatrix[pos][i] = 0.


optimizer = torch.optim.SGD(torch_model.parameters(), lr=0.01)

loss_fn = torch.nn.CrossEntropyLoss()
# loss_fn = torch.nn.MSELoss()

log_freq = 500
loss_log = []

lr_schedule = {
    0:        0.1,
    10_000:   0.01
}

current_lr = 0.001

accum_loss = 0

for i in range(1000):
    # Every data instance is an input + label pair
    data, labels = gen_data_tokens_pair()

    if i % log_freq == 0:
        print(f"At step {i}")

    # Make predictions for this batch
    outputs = torch_model(data, mask, scale, posembeddingmatrix)

    # Loss function expects labels in the form (Batch size, # Classes, other dimensions...)
    train_output = outputs[1].permute((0, 2, 1))

    # Compute the loss and its gradients
    loss = loss_fn(train_output, labels)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()
    loss_log.append(loss.item())

    if i in lr_schedule:
        optimizer = torch.optim.SGD(torch_model.parameters(), lr=lr_schedule[i])


print(loss_log[-1])

plt.plot(loss_log)
plt.show()

data, label = gen_data_tokens_pair(new_batch_size=2)
print("Data:")
print(data)
print("Label:")
print(label)
outputs = torch_model(data, mask[:2, :, :].reshape((2, bindings['ntokens'], bindings['ntokens'])), scale, posembeddingmatrix[:2, :, :].reshape((2, bindings['ntokens'], bindings['dmodel'])))
print("Outputs:")
print(outputs[0])