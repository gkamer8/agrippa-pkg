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

agrippa.export(proj_name, onnx_out)

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
             print_freq=1000,
             lr=5e-5,
             sparsity=.7,
             log_freq=100):
    
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
print(importance)
final_params = optimize(torch_model, importance)

print(f"Final W:")
w1_name = [x for x in final_params if x.split(".")[1].split("_")[0]=='W1'][0]
wt_name = [x for x in final_params if x.split(".")[1].split("_")[0]=='WT1'][0]
print(final_params[w1_name])
print(final_params[wt_name])

print("Test x:")

test_x = torch.tensor([[.7], [.0], [.3]])
print(test_x)
out = torch_model(test_x)[0]
print(out)

error = (importance*(test_x.abs() - out)**2)

loss = error.sum()
print(loss.item())

plt.plot(losses)
plt.show()