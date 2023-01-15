import sys
sys.path.append('../../src')  # where the agrippa source code is stored

import agrippa
import numpy as np
import os
import torch

proj_folder = "constant-proj"
onnx_file = "testing.onnx"

# Make sure there's no weights file
try:
    os.remove(os.path.join(proj_folder, agrippa.WEIGHTS_FNAME))
except:
    print("removal failed")

bindings = {
    'val': 10
}

# Convert xml to onnx
agrippa.export(proj_folder, onnx_file, bindings=bindings, suppress=True)

# Attempt to run
x = torch.rand((3, 1)).float()

torch_model = agrippa.onnx_to_torch(onnx_file)

y = torch_model(x)[0]

optimizer = torch.optim.SGD(torch_model.parameters(), lr=0.02, momentum=0.9)

loss_fn = torch.nn.MSELoss()

# Pretend the train, ensure the thing is constant

for _ in range(1000):
    # Every data instance is an input + label pair

    # Zero your gradients for every batch!
    optimizer.zero_grad()

    # Make predictions for this batch
    outputs = torch_model(x)[1]

    # Compute the loss and its gradients
    loss = loss_fn(outputs, x)
    loss.backward()

    # Adjust learning weights
    optimizer.step()

y1_2 = torch_model(x)[1]
y2 = torch_model(x)[0]

def approx_equal(a, b):
    eps = 0.001
    x = torch.abs(a - b)
    for i in range(len(x)):
        for k in range(len(x[i])):
            if x[i][k] > eps:
                return False
    return True

# y needs to be same as y2
# y1_2 needs to be very close to x

assert(approx_equal(y1_2, x))
assert(approx_equal(y, y2))

print("Test 8 passed")