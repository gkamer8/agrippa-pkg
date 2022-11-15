import math
import sys
sys.path.append('../../src')  # where the agrippa source code is stored

import agrippa
import onnxruntime as ort
import numpy as np
import os
import random
import math

proj_folder = "trans-proj"
onnx_fname = "decoder.onnx"

bindings = {
    'ntokens': 3,
    'nvocab': 128,
    'dmodel': 64,
    'nlayers': 4,
    'dffnhidden': 1024
}

# Convert xml to onnx
agrippa.export(proj_folder, onnx_fname, bindings=bindings, reinit=True)

print("Exported")

mask = np.ones((bindings['ntokens'], bindings['ntokens'])).astype('float32')
scale = np.array([math.sqrt(bindings['dmodel'])]).astype('float32')

# Random sequence of tokens in one hot vector matrix (column vectors)
rand_probs = np.random.random((bindings['ntokens'], bindings['nvocab']))
maxes = np.argmax(rand_probs, axis=1)

tokens = np.zeros((bindings['ntokens'], bindings['nvocab']))
tokens[np.arange(bindings['ntokens']), maxes] = 1.
tokens = tokens.astype("float32")

ort_sess = ort.InferenceSession(onnx_fname, providers=['CPUExecutionProvider'])
outputs = ort_sess.run(None, {'tokens': tokens, 'mask': mask, 'scale': scale})

print(outputs)