import os
import pandas as pd
from csv import QUOTE_NONE
import math
import sys
sys.path.append('../../../src')  # where the agrippa source code is stored

import agrippa
import onnxruntime as ort
import numpy as np
import os
import math
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from data_constants import BATCH_SIZE, SEQ_LENGTH, bos_token, device, sum_token
from preprocess import load_data, get_str_from_ids, tokenizer
import torch.nn
import pickle

proj_folder = "model"

# Base model from vaswani et al
bindings = {
    'ntokens': SEQ_LENGTH,
    'nvocab': 50258,  # GPT tokenizer is 50257, ADD ONE FOR SUMMARIZE TOKEN
    'dmodel': 512,  # 1024 in Transformer Big, 512 in base
    'dffnhidden': 2048,  # 4096 in Transformer Big, 2048 in base
    'dvalues': 64,  # 64 in Transformer Big, same in base
    'dqueries': 64,  # 64 in Transformer Big, same in base
    'dkeys': 64,  # 64 in Transformer Big, same in base
    'nheads': 8,  # 16 in Transformer Big, 8 in base
    'nlayers': 6,  # 6 in Transformer Big, same in base
}

constant_bindings = {
    'ln_eps': 1e-5,
    'scale': math.sqrt(bindings['dkeys']),
    'embed_scale': math.sqrt(bindings['dmodel']),
}

bindings.update(constant_bindings)

def get_posembeddings(isDecoder=False):
    posembeddingmatrix = torch.empty((bindings['ntokens'], bindings['dmodel']))

    for pos in range(len(posembeddingmatrix)):
        for i in range(len(posembeddingmatrix[0])):

            real_pos = pos
            # In decoder, the last one is the document embedding, which should be embedded as being first
            if isDecoder:
                if pos == len(posembeddingmatrix) - 1:
                    real_pos = 0
                else:
                    real_pos += 1

            # Even indices get one thing, odd the other
            if i % 2 == 0:
                posembeddingmatrix[pos, i] = math.sin(real_pos/(10_000**(i/bindings['dmodel'])))
            else:
                posembeddingmatrix[pos, i] = math.cos(real_pos/(10_000**(i/bindings['dmodel'])))

    posembeddingmatrix = posembeddingmatrix.to(device)
    return posembeddingmatrix


def process_sentence(token_ids):        
    # Pad by repeating
    # The plus 2 is for BOS and SUM 
    need_to_pad = SEQ_LENGTH - (len(sentence1_tokens) + 2)
    assert(need_to_pad > 0)

    padded_tokens = (sentence1_tokens * SEQ_LENGTH)[:SEQ_LENGTH-2]
    
    padded_tokens = [bos_token] + padded_tokens + [sum_token]
    assert(len(padded_tokens) == SEQ_LENGTH)

    token_tensor = torch.tensor(padded_tokens)
    encoder_tokens = F.one_hot(token_tensor, num_classes=bindings['nvocab']).float()
    return encoder_tokens

if __name__ == '__main__':

    reinit_model = False
    re_export = True

    if re_export:
        agrippa.export(proj_folder, "encoder.onnx", index="inference_enc.agr", bindings=bindings, reinit=reinit_model, suppress=False)

    enc_ort_sess = ort.InferenceSession('encoder.onnx', providers=['CPUExecutionProvider'])

    # For encoder
    zeros_mask = torch.full((bindings['ntokens'], bindings['ntokens']), 0.).to(device)
    enc_posembedmatrix = get_posembeddings()

    csv_path = os.path.join("eval-data", "sts-test.csv")

    data = pd.read_csv(csv_path, sep='\t', usecols=[4, 5, 6], header=None, quoting=QUOTE_NONE)

    target_score = data[4].tolist()
    sentences1 = data[5].tolist()
    sentences2 = data[6].tolist()

    assert(len(sentences1) == len(sentences2) == len(target_score))

    sim_scores = []

    for i in range(len(sentences1)):

        if i % 100 == 0:
            print(f"At i = {i}/{len(sentences1)}")

        sentence1_tokens = tokenizer(sentences1[i])['input_ids']
        sentence2_tokens = tokenizer(sentences2[i])['input_ids']

        encoder_tokens1 = process_sentence(sentence1_tokens)
        encoder_tokens2 = process_sentence(sentence2_tokens)

        encoder_out1 = enc_ort_sess.run(None, {'encoder_tokens': encoder_tokens1.cpu().detach().numpy(),
                                        'encoder_mask': zeros_mask.cpu().detach().numpy(),
                                        'enc_posembedmatrix': enc_posembedmatrix.cpu().detach().numpy()})[0]

        encoder_out2 = enc_ort_sess.run(None, {'encoder_tokens': encoder_tokens2.cpu().detach().numpy(),
                                        'encoder_mask': zeros_mask.cpu().detach().numpy(),
                                        'enc_posembedmatrix': enc_posembedmatrix.cpu().detach().numpy()})[0]

        embedding1 = torch.tensor(encoder_out1[-1])
        embedding2 = torch.tensor(encoder_out2[-1])

        # Calculate cosine similarity

        x = torch.dot(embedding1, embedding2).item()
        len1 = torch.sum(torch.dot(embedding1, embedding1)).item()
        len2 = torch.sum(torch.dot(embedding2, embedding2)).item()
        
        similarity = x / (len1 * len2)
        sim_scores.append(similarity)

    plt.scatter(target_score, sim_scores)
    plt.show()

    with open("results.pkl", "wb") as fhand:
        pickle.dump([target_score, sim_scores], fhand)
