import sys
sys.path.append('../../src')  # where the agrippa source code is stored

import agrippa
import onnxruntime as ort
import numpy as np
import os

# Make sure there's no weights file
try:
    os.remove(os.path.join('default-project', agrippa.WEIGHTS_FNAME))
except:
    print("removal failed")

# Convert xml to onnx
agrippa.export('default-project', 'testing.onnx', suppress=True)

# Attempt to run
x = np.random.random((3, 1)).astype("float32")
ort_sess = ort.InferenceSession('testing.onnx')
outputs = ort_sess.run(None, {'input': x})

# Convert again (should be using the same weights)
agrippa.export('default-project', 'testing.onnx', suppress=True)
# Attempt to run
ort_sess = ort.InferenceSession('testing.onnx')
outputs2 = ort_sess.run(None, {'input': x})

np.testing.assert_array_equal(outputs,
                              outputs2)

print("Test 1 passed")