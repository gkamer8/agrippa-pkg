import sys
sys.path.append('../../src')  # where the agrippa source code is stored

import agrippa
import onnxruntime as ort
import numpy as np

# Conver xml to onnx
agrippa.export('default.xml', 'testing.onnx')

# Attempt to run
x = np.random.random((3, 1)).astype("float32")
ort_sess = ort.InferenceSession('testing.onnx')
outputs = ort_sess.run(None, {'input': x})
