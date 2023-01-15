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

batch_size = 1
seq_length = 64

# To make things easier when we need to use a smaller batch
effective_batch_size = batch_size // 1

proj_folder = "model"
onnx_fname = "transformer.onnx"

# Base model from vaswani et al
bindings = {
    'ntokens': seq_length,
    'nvocab': 50257,
    'dmodel': 1024,
    'dffnhidden': 4096,
    'dvalues': 64,
    'dqueries': 64,
    'dkeys': 64,
    'nheads': 3,  # 16 in Transformer Big
    'nlayers': 2,  # 6 in Transformer Big
}

constant_bindings = {
    'ln_eps': 1e-5,
    'scale': math.sqrt(bindings['dkeys']),
    'embed_scale': math.sqrt(bindings['dmodel'])
}

bindings.update(constant_bindings)

# Convert xml to onnx
agrippa.export(proj_folder, onnx_fname, index="encoder.agr", bindings=bindings, reinit=True, suppress=True)

print("Exported")

device = "cuda:0"

# Now we train
torch_model = agrippa.onnx_to_torch(onnx_fname)
torch_model = torch_model.to(device)
