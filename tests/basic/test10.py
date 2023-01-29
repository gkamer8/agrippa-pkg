import sys
from unicodedata import bidirectional
sys.path.append('../../src')  # where the agrippa source code is stored

import agrippa
import onnxruntime as ort
import numpy as np
import os

"""

Test searching for weights

"""

proj_name = "import-nest-proj"

bindings = {
    'n': 5
}

# Just confirm that the thing exists
# Convert xml to onnx
agrippa.export(proj_name, 'testing.onnx', bindings=bindings, index="main.agr", suppress=True)

matches = agrippa.utils.find_params("B", proj_name)
assert(len(matches) == 7)

print("Test 10 passed")