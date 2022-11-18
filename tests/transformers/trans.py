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


proj_folder = "trans-proj"
onnx_fname = "decoder.onnx"

bindings = {
    'ntokens': 3,
    'nvocab': 128,
    'dmodel': 12,
    'nlayers': 4,
    'dffnhidden': 32
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

def gen_data_tokens_pair(a_first=None):
    label = torch.zeros((bindings['ntokens'])).long()
    data = torch.zeros((bindings['ntokens'], bindings['nvocab']))
    if a_first is None:
        a_first = random.random() < .5
    if a_first:
        label[0] = 0  # a
        data[0][127] = 1. # <BOS>

        label[1] = 1  # b
        data[1][0] = 1.   # a

        label[2] = 2  # c
        data[2][1] = 1.   # b
    else:
        label[0] = 3  # d
        data[0][127] = 1. # <BOS>

        label[1] = 4  # e
        data[1][3] = 1.   # d

        label[2] = 5  # f
        data[2][4] = 1.   # e

    """label = torch.zeros((bindings['ntokens'], bindings['nvocab']))
    data = torch.zeros((bindings['ntokens'], bindings['nvocab']))
    if a_first is None:
        a_first = random.random() < .5
    if a_first:
        label[0][0] = 1.  # a
        data[0][127] = 1. # <BOS>

        label[1][1] = 1.  # b
        data[1][0] = 1.   # a

        label[2][2] = 1.  # c
        data[2][1] = 1.   # b
    else:
        label[0][3] = 1.  # d
        data[0][127] = 1. # <BOS>

        label[1][4] = 1.  # e
        data[1][3] = 1.   # d

        label[2][5] = 1.  # f
        data[2][4] = 1.   # e"""
    
    return data, label

scale = math.sqrt(bindings['dmodel'])
mask = torch.tensor([
    [0, -float("inf"), -float("inf")],
    [0, 0, -float("inf")],
    [0, 0, 0]
])

# Straight from Vaswani et al
posembeddingmatrix = torch.empty((bindings['ntokens'], bindings['dmodel']))
for pos in range(len(posembeddingmatrix)):
    for i in range(len(posembeddingmatrix[0])):
        # Even indices get one thing, odd the other
        if i % 2 == 0:
            posembeddingmatrix[pos][i] = math.sin(pos/(10_000**(i/bindings['dmodel'])))
            # posembeddingmatrix[pos][i] = 0.
        else:
            posembeddingmatrix[pos][i] = math.cos(pos/(10_000**(i/bindings['dmodel'])))
            # posembeddingmatrix[pos][i] = 0.


optimizer = torch.optim.SGD(torch_model.parameters(), lr=0.01)

loss_fn = torch.nn.CrossEntropyLoss()
# loss_fn = torch.nn.MSELoss()

batch_size = 25

log_freq = 500
loss_log = []

lr_schedule = {
    0:        0.01,
    10_000:   0.005
}

current_lr = 0.001

accum_loss = 0

for i in range(batch_size * 1000):
    # Every data instance is an input + label pair
    data, labels = gen_data_tokens_pair()

    if i % log_freq == 0:
        print(f"At step {i}")

    # Make predictions for this batch
    outputs = torch_model(data, mask, scale, posembeddingmatrix)

    # Compute the loss and its gradients
    loss = loss_fn(outputs[1], labels) / batch_size
    loss.backward()
    accum_loss += loss.item()

    # Zero your gradients for every batch!
    if i != 0 and i % batch_size == 0:
        optimizer.step()
        optimizer.zero_grad()
        loss_log.append(accum_loss)
        accum_loss = 0
        if i >= 30_000 and i <= 30_100:
            with open("model_weights" + str(i) + ".txt", "w") as fhand:
                fhand.write(str(torch_model.state_dict()))

    if i in lr_schedule:
        optimizer = torch.optim.SGD(torch_model.parameters(), lr=lr_schedule[i])


print(loss_log[-1])

plt.plot(loss_log)
plt.show()

data, label = gen_data_tokens_pair(a_first=True)
print("Data:")
print(data)
print("Label:")
print(label)
outputs = torch_model(data, mask, scale, posembeddingmatrix)
print("Outputs:")
print(outputs[0])

data, label = gen_data_tokens_pair(a_first=False)
print("Data:")
print(data)
print("Label:")
print(label)
outputs = torch_model(data, mask, scale, posembeddingmatrix)
print("Outputs:")
print(outputs[0])