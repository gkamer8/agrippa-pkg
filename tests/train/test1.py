import sys
sys.path.append('../../src')  # where the agrippa source code is stored

import numpy as np
import agrippa

import torch
# from agrippa import onnx2torch as ot

proj_name = 'simple-project'
onnx_out = 'simple_testing.onnx'

agrippa.export(proj_name, onnx_out)

torch_model = agrippa.onnx_to_torch(onnx_out)

x = torch.rand((3, 1))

out_torch = torch_model(x)
print(f"Out: {out_torch}")

optimizer = torch.optim.SGD(torch_model.parameters(), lr=0.001, momentum=0.9)

loss_fn = torch.nn.MSELoss()

pairs = []

for _ in range(100):
    x = torch.rand((3, 1))
    y = (x * 2) + 5
    pairs.append((x, y))

# Here, we use enumerate(training_loader) instead of
# iter(training_loader) so that we can track the batch
# index and do some intra-epoch reporting
for _ in range(100):
    for i, data in enumerate(pairs):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = torch_model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
print(torch_model.state_dict())

print("Test 1 passed")