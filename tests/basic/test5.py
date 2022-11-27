import sys
from unicodedata import bidirectional
sys.path.append('../../src')  # where the agrippa source code is stored

import agrippa
import onnxruntime as ort
import numpy as np
import os

"""

Test the expression functionality

"""

proj_name = "expr-project"

# Make sure there's no weights file
try:
    os.remove(os.path.join(proj_name, agrippa.WEIGHTS_FNAME))
except:
    print("removal failed")

# Convert xml to onnx
bindings = {
    'bound1': 2,
    'bound2': 3
}
agrippa.export(proj_name, 'testing.onnx', bindings=bindings, suppress=True)

x = np.random.random((6, 3)).astype("float32")
x2 = np.random.random((bindings['bound1']*bindings['bound2'], bindings['bound2'])).astype("float32")
ort_sess = ort.InferenceSession('testing.onnx', providers=['CPUExecutionProvider'])
outputs = ort_sess.run(None, {'input': x, 'input2': x2})

# For first output
rows = len(outputs[0])
cols = len(outputs[0][0])

assert(rows == 6)
assert(cols == 3)

# For second output
rows = len(outputs[1])
cols = len(outputs[1][0])

assert(rows == bindings['bound1']*bindings['bound2'])
assert(cols == bindings['bound2'])

print("Test 5 passed")