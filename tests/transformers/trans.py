import sys
sys.path.append('../../src')  # where the agrippa source code is stored

import agrippa
import onnxruntime as ort
import numpy as np
import os
import random

proj_folder = "trans-proj"
onnx_fname = "decoder.onnx"

bindings = {
    'ntokens': 3,
    'nvocab': 128,
    'dmodel': 64,
    'nlayers': 4
}

# Convert xml to onnx
agrippa.export(proj_folder, onnx_fname, bindings=bindings)

# Random sequence of tokens in one hot vector matrix (column vectors)
rand_probs = np.random.random((bindings['nvocab'], bindings['ntokens']))
maxes = np.argmax(rand_probs, axis=0)
tokens = np.zeros((bindings['nvocab'], bindings['ntokens']))
tokens[maxes, np.arange(bindings['ntokens'])] = 1.
tokens = tokens.astype("float32")

ort_sess = ort.InferenceSession(onnx_fname, providers=['CPUExecutionProvider'])
outputs = ort_sess.run(None, {'tokens': tokens})

print(outputs)