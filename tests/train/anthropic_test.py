import sys
sys.path.append('../../src')  # where the agrippa source code is stored

import numpy as np
import agrippa
import onnxruntime as ort

import torch

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

def optimize(model,
             importance,
             steps=100,
             print_freq=10,
             lr=1e-3,
             sparsity=.5):
    
    opt = torch.optim.AdamW(list(model.parameters()), lr=lr)

    for step in range(steps):
        batch = generate_input(features, sparsity)

        out = model(batch)[0]  # recall first is final output, second is intermediate output

        error = (importance*(batch.abs() - out)**2)

        loss = error.sum()
        loss.backward()

        opt.step()

        if step % print_freq == 0:
            print(f"At step {step+1}/{steps}")
    
    print(f"Final parameters: {model.state_dict()}")


def gen_importance(exp=.5):
    # Expotential feature importance
    importance = exp**torch.arange(features)
    return importance

importance = gen_importance()
optimize(torch_model, importance)