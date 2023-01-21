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
from data_constants import BATCH_SIZE, SEQ_LENGTH, bos_token, device
from preprocess import load_data, get_str_from_ids
import torch.nn

proj_folder = "model"
onnx_fname = "transformer.onnx"

# Base model from vaswani et al
bindings = {
    'ntokens': SEQ_LENGTH,
    'nvocab': 50257,
    'dmodel': 512,  # 1024 in Transformer Big, 512 in base
    'dffnhidden': 2048,  # 4096 in Transformer Big, 2048 in base
    'dvalues': 64,  # 64 in Transformer Big, same in base
    'dqueries': 64,  # 64 in Transformer Big, same in base
    'dkeys': 64,  # 64 in Transformer Big, same in base
    'nheads': 8,  # 16 in Transformer Big, 8 in base
    'nlayers': 6,  # 6 in Transformer Big, same in base
}

constant_bindings = {
    'ln_eps': 1e-5,
    'scale': math.sqrt(bindings['dkeys']),
    'embed_scale': math.sqrt(bindings['dmodel'])
}

bindings.update(constant_bindings)

def save_model():
    weights_dict = {}
    whole_dict = torch_model.state_dict()
    for key in whole_dict:
        real_name = key[len('initializers.'):]
        weights_dict[real_name] = whole_dict[key]

    with open(os.path.join("model", 'weights.pkl'), 'wb') as fhand:
        pickle.dump(weights_dict, fhand)

if __name__ == '__main__':

    reinit_model = True
    # Convert xml to onnx
    agrippa.export(proj_folder, onnx_fname, index="transformer.agr", bindings=bindings, reinit=reinit_model, suppress=True)
    print("Exported")

    # Now we train
    torch_model = agrippa.onnx_to_torch(onnx_fname)

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

    zeros_mask = torch.full((BATCH_SIZE, bindings['ntokens'], bindings['ntokens']), 0.).to(device)

    # Vaswani et al use label smoothing = 0.1
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    # loss_fn = torch.nn.MSELoss()

    log_freq = 1
    loss_log = []

    current_lr = 0
    optimizer = torch.optim.Adam(torch_model.parameters(), lr=current_lr, betas=(0.9, 0.98), eps=10e-09)
    # optimizer = torch.optim.SGD(torch_model.parameters(), lr=current_lr)

    accum_loss = 0

    max_train_size = 100_000  # max total for an epoch

    gradient_accum_steps = 2

    warmup_steps = 4000

    start_buffer = 0

    nepochs = 3

    save_batch_freq = 100

    optim_steps = 1
    for epoch in range(nepochs):
        print(f"Epoch {epoch}")
        counter = 0
        new_lr = 0.01
        for i, (other, english) in enumerate(load_data()):

            if i % log_freq == 0:
                print(f"At step {i}")

            # LR With warmup
            new_lr = bindings['dmodel'] ** (-.5) * min((start_buffer+optim_steps)**(-0.5), (start_buffer+optim_steps)*warmup_steps**(-1.5))
            for g in optimizer.param_groups:
                g['lr'] = new_lr
            
            # Add dimension to data

            english_data = F.one_hot(english.to(torch.int64)).float()

            # (Batch, Seq length)
            chopped = other[:, :-1]
            to_cat = torch.full((BATCH_SIZE, 1), bos_token).to(device)
            other_data = torch.cat((to_cat, chopped), -1).to(torch.int64)
            other_data = F.one_hot(other_data.to(torch.int64)).float()

            # Make predictions for this batch
            # (decoder tokens, encoder tokens, decoder mask, encoder mask, pos embedding matrix)
            outputs = torch_model(other_data, english_data, mask, zeros_mask, posembeddingmatrix).to(device)
            
            # Loss function expects labels in the form (Batch size, # Classes, other dimensions...)
            train_output = outputs.permute((0, 2, 1))

            # Compute the loss and its gradients
            loss = loss_fn(train_output, other.to(torch.int64)) / gradient_accum_steps
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
