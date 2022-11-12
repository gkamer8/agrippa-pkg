import sys
sys.path.append('../../src')  # where the agrippa source code is stored

import numpy as np
import agrippa
import onnxruntime as ort

import torch
import matplotlib.pyplot as plt

# Recreate the anthropic thing sort of

proj_name = 'anthropic-proj'
onnx_out = 'anthropic.onnx'

agrippa.export(proj_name, onnx_out, reinit=True)

torch_model = agrippa.onnx_to_torch(onnx_out)

def generate_input(features, feature_probability):
    feat = torch.rand((features, 1))
    batch = torch.where(
        torch.rand((features, 1)) <= feature_probability,
        feat,
        torch.zeros(()),
    )
    return batch

features = 3

losses = []

def optimize(model,
             importance,
             steps=100_000,
             print_freq=10_000,
             lr=5e-5,
             sparsity=.7,
             log_freq=1_000):
    
    opt = torch.optim.AdamW(list(model.parameters()), lr=lr)

    for step in range(steps):
        opt.zero_grad()

        batch = generate_input(features, sparsity)
        out = model(batch)[0]  # recall first is final output, second is intermediate output

        error = (importance*(batch.abs() - out)**2)

        loss = error.sum()
        loss.backward()

        opt.step()

        if step % print_freq == 0:
            print(f"At step {step+1}/{steps}")

        if step % log_freq == 0:
            losses.append(loss.item())
    
    return model.state_dict()


def gen_importance(exp=.5):
    # Expotential feature importance
    importance = torch.tensor([[exp**i] for i in range(features)])
    return importance

importance = gen_importance()


final_params = {}
if True:  # large
    final_params = optimize(torch_model, importance)
else:  # small
    final_params = optimize(torch_model, importance, steps=100, print_freq=20, log_freq=1)

print("W:")
print(final_params['initializers.W'])

print("(WT)(W):")
print(torch.matmul(torch.transpose(final_params['initializers.W'], 0, 1), final_params['initializers.W']))

print("b:")
print(final_params['initializers.b'])

print("Test x:")

test_x = torch.tensor([[.7], [.0], [.3]])
print(test_x)
out = torch_model(test_x)[0]
print("out:")
print(out)

error = (importance*(test_x.abs() - out)**2)

loss = error.sum()
print("Loss:")
print(loss.item())

plt.plot(losses)
plt.show()

x1 = final_params['initializers.W'][0, 0]
y1 = final_params['initializers.W'][1, 0]

x2 = final_params['initializers.W'][0, 1]
y2 = final_params['initializers.W'][1, 1]

plt.plot([0, x1], [0, y1])
plt.plot([0, x2], [0, y2])
plt.xlim(-1, 1)
plt.ylim(-1, 1)

plt.show()