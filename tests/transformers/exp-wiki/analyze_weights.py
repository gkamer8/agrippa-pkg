import matplotlib.pyplot as plt
import os
import pickle

weights_dict = {}
with open(os.path.join('model', 'weights.pkl'), 'rb') as fhand:
    weights_dict = pickle.load(fhand)


# Assumption: weights are at most two dimensional
param_num = 0
for key in weights_dict:
    if len(weights_dict[key].shape) < 2:
        param_num += weights_dict[key].shape[0]
    else:
        param_num += weights_dict[key].shape[-1] * weights_dict[key].shape[-2]

print(f"Number of parameters: {param_num}")