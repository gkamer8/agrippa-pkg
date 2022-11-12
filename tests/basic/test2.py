import sys
sys.path.append('../../src')  # where the agrippa source code is stored

import agrippa
import onnxruntime as ort
import numpy as np
import os

"""

Test the transpose operator for onnx conversion

"""

proj_folder = "transpose-proj"
onnx_output = "testing.onnx"

# Make sure there's no weights file
try:
    os.remove(os.path.join(proj_folder, agrippa.WEIGHTS_FNAME))
except:
    print("removal failed")

# Convert xml to onnx
agrippa.export(proj_folder, onnx_output, suppress=True)

# The agr file is supposed to do a transpose with perm (0, 1, 3, 2)

# Attempt to run
x = np.random.random((3, 3, 3, 3)).astype("float32")
ort_sess = ort.InferenceSession(onnx_output, providers=['CPUExecutionProvider'])
outputs = ort_sess.run(None, {'input': x})

xtran = np.transpose(x, axes=(0, 1, 3, 2))

np.testing.assert_array_equal(xtran,
                              outputs[0])

print("Test 2 passed")