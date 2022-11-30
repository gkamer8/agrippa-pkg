import sys
from unicodedata import bidirectional
sys.path.append('../../src')  # where the agrippa source code is stored

import agrippa
import onnxruntime as ort
import numpy as np
import os

"""

Test stretching

"""

proj_name = "stretch-proj"

# Make sure there's no weights file
try:
    os.remove(os.path.join(proj_name, agrippa.WEIGHTS_FNAME))
except:
    print("removal failed")

# Convert xml to onnx
agrippa.export(proj_name, 'testing.onnx', suppress=True)

x = np.random.random((3, 2)).astype("float32")
ort_sess = ort.InferenceSession('testing.onnx', providers=['CPUExecutionProvider'])
outputs = ort_sess.run(None, {'input': x})

print(outputs[0])

print("Test 6 passed")