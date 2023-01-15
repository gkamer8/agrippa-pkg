import sys
from unicodedata import bidirectional
sys.path.append('../../src')  # where the agrippa source code is stored

import agrippa
import onnxruntime as ort
import numpy as np
import os

"""

Test importing from other Agrippa file

"""

proj_name = "import-proj"

# Make sure there's no weights file
try:
    os.remove(os.path.join(proj_name, agrippa.WEIGHTS_FNAME))
except:
    print("removal failed")

bindings = {
    'n': 5
}

# Convert xml to onnx
agrippa.export(proj_name, 'testing.onnx', bindings=bindings, index="main.agr", suppress=True)

x = np.random.random((bindings['n'], 1)).astype("float32")
ort_sess = ort.InferenceSession('testing.onnx', providers=['CPUExecutionProvider'])
outputs = ort_sess.run(None, {'half1$input': x})

print(outputs[0])

print("Test 7 passed")