import sys
sys.path.append('../../src')  # where the agrippa source code is stored

import agrippa
import onnxruntime as ort
import numpy as np
import os

"""

Test the Softmax operation

"""

proj_name = "softmax-proj"

# Make sure there's no weights file
try:
    os.remove(os.path.join(proj_name, agrippa.WEIGHTS_FNAME))
except:
    print("removal failed")

# Convert xml to onnx
agrippa.export(proj_name, 'testing.onnx', suppress=True)

# Attempt to run
x = np.ones((10, 2)).astype("float32")
ort_sess = ort.InferenceSession('testing.onnx', providers=['CPUExecutionProvider'])
outputs = ort_sess.run(None, {'input': x})

# Convert again (should be using the same weights)
agrippa.export(proj_name, 'testing.onnx', suppress=True)
# Attempt to run
ort_sess = ort.InferenceSession('testing.onnx', providers=['CPUExecutionProvider'])
outputs2 = ort_sess.run(None, {'input': x})

np.testing.assert_array_equal(outputs,
                              outputs2)

# Ensures the softmax was along the correct axis
assert(outputs[0][0][0] < .4)

print("Test 3 passed")