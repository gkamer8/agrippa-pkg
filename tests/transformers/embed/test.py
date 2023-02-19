import pickle
import math
import sys
sys.path.append('../../../src')  # where the agrippa source code is stored

import agrippa
import onnxruntime as ort
import numpy as np
import os
import random
import math
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from data_constants import BATCH_SIZE, SEQ_LENGTH, bos_token, device, sum_token
from preprocess import load_data, get_str_from_ids
import torch.nn

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

if __name__ == '__main__':

    reinit_model = False
    re_export = False

    if re_export:
        # Export inference decoder
        agrippa.export(proj_folder, "decoder.onnx", index="inference_dec.agr", bindings=bindings, reinit=reinit_model, suppress=False)
        agrippa.export(proj_folder, "encoder.onnx", index="inference_enc.agr", bindings=bindings, reinit=reinit_model, suppress=False)

    dec_ort_sess = ort.InferenceSession('decoder.onnx', providers=['CPUExecutionProvider'])
    enc_ort_sess = ort.InferenceSession('encoder.onnx', providers=['CPUExecutionProvider'])

    # For encoder
    zeros_mask = torch.full((bindings['ntokens'], bindings['ntokens']), 0.).to(device)
    enc_posembedmatrix = get_posembeddings()

    # For decoder
    encoder_output_mask = torch.full((bindings['ntokens'], bindings['dmodel']), 0.)
    encoder_output_mask[-1, :] = 1.
    encoder_output_mask = encoder_output_mask.to(device)
    decoder_embed_removal_mask = torch.full((bindings['ntokens'], bindings['dmodel']), 1.)
    decoder_embed_removal_mask[-1, :] = 0.
    decoder_embed_removal_mask = decoder_embed_removal_mask.to(device)

    proto_mask = torch.full((bindings['ntokens'], bindings['ntokens']), -float("inf"))
    proto_mask = torch.triu(proto_mask, diagonal=1)    
    proto_mask[:, -1] = 0.  # in decoder, every token can see the last one (doc embedding)
    proto_mask[-1, :-1] = -float("inf")  # but the doc embedding can only see itself
    decoder_mask = proto_mask.to(device)

    dec_posembedmatrix = get_posembeddings(isDecoder=True)

    for i, data in enumerate(load_data(split="test", batch_size=2)):

        use_batch_index = 0  # which example in the batch are we using

        print("Example:")
        print(get_str_from_ids(data[use_batch_index]))

        chopped = data[use_batch_index, :-1]
        to_cat = torch.full((1,), bos_token).to(device)
        right_shifted_ids = torch.cat((to_cat, chopped), -1).to(torch.int64)
        right_shifted = F.one_hot(right_shifted_ids.to(torch.int64), num_classes=bindings['nvocab']).float()
        # Add summarize token to end
        encoder_chopped = right_shifted_ids[:-1]
        to_cat = torch.full((1,), sum_token).to(device)
        encoder_tokens_ids = torch.cat((encoder_chopped, to_cat), -1).to(torch.int64)
        encoder_tokens = F.one_hot(encoder_tokens_ids.to(torch.int64), num_classes=bindings['nvocab']).float()

        """

        # Decoder inputs:

        <import from="decoder_tokens" dim="[var(ntokens), var(nvocab)]" />
        <import from="encoder_output" dim="[var(ntokens), var(dmodel)]" />

        <import from="decoder_mask" dim="[var(ntokens), var(ntokens)]" />

        <import from="dec_posembedmatrix" dim="[var(ntokens), var(dmodel)]" />

        <import from="encoder_output_mask" dim="[var(ntokens), var(dmodel)]" />
        <import from="decoder_embed_removal_mask" dim="[var(ntokens), var(dmodel)]" />
        
        # Decoder outputs:

        <export from="end_linear_decoder" dim="[var(ntokens), var(nvocab)]" />

        """

        """

        # Encoder inputs:

        <import from="encoder_tokens" dim="[var(ntokens), var(nvocab)]" />
        <import from="encoder_mask" dim="[var(ntokens), var(ntokens)]" />
        <import from="enc_posembedmatrix" dim="[var(ntokens), var(dmodel)]" />

        # Encoder outputs:
        <export from="encoder$ln2$layer_norm_out" dim="[var(ntokens), var(dmodel)]" />

        """

        encoder_out = enc_ort_sess.run(None, {'encoder_tokens': encoder_tokens.cpu().detach().numpy(),
                                        'encoder_mask': zeros_mask.cpu().detach().numpy(),
                                        'enc_posembedmatrix': enc_posembedmatrix.cpu().detach().numpy()})[0]

        decoder_out = dec_ort_sess.run(None, {'decoder_tokens': right_shifted.cpu().detach().numpy(),
                                        'encoder_output': encoder_out,
                                        'decoder_mask': decoder_mask.cpu().detach().numpy(),
                                        'dec_posembedmatrix': dec_posembedmatrix.cpu().detach().numpy(),
                                        'encoder_output_mask': encoder_output_mask.cpu().detach().numpy(),
                                        'decoder_embed_removal_mask': decoder_embed_removal_mask.cpu().detach().numpy()
                                        })[0]

        decoder_out = torch.tensor(decoder_out)

        print(decoder_out)
        print(decoder_out.shape)

        topk = torch.topk(decoder_out, k=10, dim=1)[1]  # gets indices of top k in tensor of shape (seq length, 2)
        tops = topk[:, 0].flatten()

        print(tops.shape)

        print("(actual) : (prediction)")
        for i in range(SEQ_LENGTH):
            actual = get_str_from_ids([data[use_batch_index][i]])
            predicted = get_str_from_ids([tops[i]])
            txt = f"({actual}) : ({predicted})"
            print(txt)

        exit(0)
