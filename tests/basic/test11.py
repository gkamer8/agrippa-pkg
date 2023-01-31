import sys
sys.path.append('../../src')  # where the agrippa source code is stored

import agrippa

"""

Test saving project weights

"""

proj_name = "import-nest-proj"

onnx_out = 'testing.onnx'

bindings = {
    'n': 5
}

agrippa.export(proj_name, 'testing.onnx', bindings=bindings, index="main.agr", suppress=True)

torch_model = agrippa.onnx_to_torch(onnx_out)

agrippa.utils.save_pytorch_model(torch_model, proj_name)

matches = agrippa.utils.find_params("B", proj_name)

# No dots to make sure we removed the "initializers."
assert(len([x for x in matches if "." in x]) == 0)
assert(len(matches) == 7)

print("Test 11 passed")