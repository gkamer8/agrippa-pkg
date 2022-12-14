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
import torch.nn.functional as F

batch_size = 200
seq_length = 64

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

# Convert xml to onnx
agrippa.export(proj_folder, onnx_fname, bindings=bindings, reinit=True, suppress=True)

print("Exported")

device = "cuda:0"

# Now we train
torch_model = agrippa.onnx_to_torch(onnx_fname)
torch_model = torch_model.to(device)

nbatches = len(os.listdir(os.path.join('dataset', 'batches')))
batch_order = [i for i in range(nbatches)]
np.random.shuffle(batch_order)
bos_token = 50256
def gen_data_tokens_pair():
    np.random.shuffle(batch_order)
    for i in range(len(batch_order)):
        with open(os.path.join('dataset', 'batches', f'{batch_order[i]}.pkl'), 'rb') as fhand:
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
        x = x.to(device)
        y = y.to(device)
        yield x, y

scale = math.sqrt(bindings['dkeys'])
embed_scale = math.sqrt(bindings['dmodel'])
ln_eps = 1e-5

proto_mask = torch.full((batch_size, bindings['ntokens'], bindings['ntokens']), -float("inf"))
proto_mask[:] = torch.triu(proto_mask[0], diagonal=1)
mask = proto_mask
mask = mask.to(device)

# Straight from Vaswani et al
posembeddingmatrix = torch.empty((batch_size, bindings['ntokens'], bindings['dmodel']))
for pos in range(len(posembeddingmatrix[0])):
    for i in range(len(posembeddingmatrix[0][0])):
        # Even indices get one thing, odd the other
        if i % 2 == 0:
            posembeddingmatrix[:, pos, i] = math.sin(pos/(10_000**(i/bindings['dmodel'])))
        else:
            posembeddingmatrix[:, pos, i] = math.cos(pos/(10_000**(i/bindings['dmodel'])))
posembeddingmatrix = posembeddingmatrix.to(device)

# Vaswani et al use label smoothing = 0.1
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
# loss_fn = torch.nn.MSELoss()

log_freq = 1
loss_log = []

current_lr = 0
optimizer = torch.optim.Adam(torch_model.parameters(), lr=current_lr, betas=(0.9, 0.98), eps=1e-09)
# optimizer = torch.optim.SGD(torch_model.parameters(), lr=current_lr)

accum_loss = 0

max_train_size = 100_000  # total for an epoch

gradient_accum_steps = 5

warmup_steps = 2000

start_buffer = 0

nepochs = 2

save_batch_freq = 200

def save_model():
    weights_dict = {}
    whole_dict = torch_model.state_dict()
    for key in whole_dict:
        real_name = key[len('initializers.'):]
        weights_dict[real_name] = whole_dict[key]

    with open(os.path.join("model", 'weights.pkl'), 'wb') as fhand:
        pickle.dump(weights_dict, fhand)

optim_steps = 1
for epoch in range(nepochs):
    print(f"Epoch {epoch}")
    counter = 0
    new_lr = 0.01
    for i, (data, label) in enumerate(gen_data_tokens_pair()):

        if i % log_freq == 0:
            print(f"At step {i}")

        # LR With warmup
        new_lr = bindings['dmodel'] ** (-.5) * min(optim_steps**(-0.5), optim_steps*warmup_steps**(-1.5))
        for g in optimizer.param_groups:
            g['lr'] = new_lr
        
        # Add dimension to data
        data = F.one_hot(data).float()

        # Make predictions for this batch
        outputs = torch_model(data, mask, scale, posembeddingmatrix, embed_scale, ln_eps)
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
            print(f"Loss: {accum_loss}")
            accum_loss = 0
            optim_steps += 1

        if counter % save_batch_freq == 0:
            print("Saving model...")
            save_model()
            print(f"Latest lr: {new_lr}")
            print("Saving logs...")
            save_log_path = os.path.join("logs", f"log_{optim_steps}.txt")
            with open(save_log_path, 'w') as fhand:
                fhand.write(str(loss_log))

        counter += 1
        if counter >= max_train_size:
            break

save_model()
print(loss_log[-1])
print(loss_log)