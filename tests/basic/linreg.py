import sys
sys.path.append('../../src')  # where the agrippa source code is stored

import agrippa
import torch
import onnxruntime as ort

"""

Test linear regression, as seen in Getting Started example

"""

proj_name = "lin-reg"

onnx_out = 'testing.onnx'

bindings = {
    'n': 5
}

agrippa.export(proj_name, 'testing.onnx', bindings=bindings, suppress=True)
torch_model = agrippa.onnx_to_torch(onnx_out)

target_weights = torch.rand((bindings['n'],))
target_biases = torch.rand((bindings['n'],))

def get_pair():
    x = torch.rand((bindings['n'],))
    y = target_weights * x + target_biases
    return x, y

optimizer = torch.optim.SGD(torch_model.parameters(), lr=0.1)

loss_fn = torch.nn.MSELoss()

for _ in range(5000):
    x, y = get_pair()

    # Zero your gradients for every batch!
    optimizer.zero_grad()

    # Make predictions for this batch
    outputs = torch_model(x)

    # Compute the loss and its gradients
    loss = loss_fn(outputs, y)
    loss.backward()

    # Adjust learning weights
    optimizer.step()

agrippa.utils.save_torch_model(torch_model, proj_name)

biases = agrippa.utils.find_params("b", proj_name)
weights = agrippa.utils.find_params("W", proj_name)

print(f"Target weights: {target_weights}")
print(f"Actual weights: {weights}")
print()
print(f"Target biases: {target_biases}")
print(f"Actual biases: {biases}")

agrippa.export(proj_name, 'testing.onnx', bindings=bindings, reinit=False)

ort_sess = ort.InferenceSession(onnx_out, providers=['CPUExecutionProvider'])

x, y = get_pair()
x = x.detach().numpy()
outputs = ort_sess.run(None, {'x': x})

print(f"Label: {y}")
print(f"Output: {outputs}")
