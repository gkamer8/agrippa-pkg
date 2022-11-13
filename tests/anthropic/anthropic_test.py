import pickle
import sys
sys.path.append('../../src')  # where the agrippa source code is stored

import numpy as np
import agrippa
import onnxruntime as ort

import torch
import matplotlib.pyplot as plt

import json

# Recreate the anthropic thing sort of

proj_name = 'anthropic-proj'
onnx_out = 'anthropic.onnx'
features = 5
bindings = {'features': str(features)}

agrippa.export(proj_name, onnx_out, reinit=True, bindings=bindings)

torch_model = agrippa.onnx_to_torch(onnx_out)

def generate_input(features, feature_probability, nbatch=1024):
    feat = torch.rand((nbatch, features, 1))
    batch = torch.where(
        torch.rand((nbatch, features, 1)) <= feature_probability,
        feat,
        torch.zeros(()),
    )
    return batch

losses = []

def optimize(model,
             importance,
             steps=10_000,
             print_freq=100,
             lr=1e-3,
             sparsity=.01,
             log_freq=10):
    
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


def gen_importance(exp=1):
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

test_x = torch.tensor([[.7], [.3]] + [[0.] for _ in range(features-2)])
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

pairs = []
for i in range(features):
    x = final_params['initializers.W'][0, i]
    y = final_params['initializers.W'][1, i]
    pairs.append((x, y))

for x, y in pairs:
    plt.plot([0, x], [0, y])

plt.xlim(-1, 1)
plt.ylim(-1, 1)

plt.show()

# output results

results_file = "anthropic_results.pkl"
with open(results_file, 'wb') as fhand:
    results = {'params': final_params, 'importance': importance, 'losses': losses}
    pickle.dump(results, fhand)