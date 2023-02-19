import pickle
import os
from preprocess import tokenizer
from data_constants import bos_token
import numpy as np
import torch

def find_weight(name, suppress=False):
    model_folder = "model"
    weight_dict = pickle.load(open(os.path.join(model_folder, "weights.pkl"), "rb"))

    for key in weight_dict:
        if name in key:
            if not suppress:
                print(weight_dict[key])
            return weight_dict[key]

if __name__ == '__main__':
    w = find_weight("EmbedW")
    print("BOS Token embedding: ")
    print(w[bos_token])
    print(torch.sum(w[bos_token].cpu()))
    print()

