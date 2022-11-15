import sys
sys.path.append('../../src')  # where the agrippa source code is stored

import agrippa
import onnxruntime as ort
import numpy as np
import os

"""

Test the LayerNorm and associated operations (ReduceMean, for one)

"""

proj_name = "layernorm"

# Make sure there's no weights file
try:
    os.remove(os.path.join(proj_name, agrippa.WEIGHTS_FNAME))
except:
    print("removal failed")

# Convert xml to onnx
agrippa.export(proj_name, 'testing.onnx', suppress=True)

x = np.ones((10,5)).astype("float32")
x[:, :2] = 4

ort_sess = ort.InferenceSession('testing.onnx', providers=['CPUExecutionProvider'])
outputs = ort_sess.run(None, {'input': x})

# All values for a column should match, though the gains/biases will cause values between columns to differ
truths = np.all(outputs[0] == outputs[0][0,:], axis = 0)
assert(truths.all())

print("Test 4 passed")