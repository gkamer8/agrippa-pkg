from datasets import load_dataset
from transformers import GPT2Tokenizer
from preprocess import load_data, get_str_from_ids
from train import bindings
from data_constants import BATCH_SIZE, SEQ_LENGTH, bos_token
import random
import torch
import sys
sys.path.append('../../../src')  # where the agrippa source code is stored
import torch.nn.functional as F
import math
import agrippa
import numpy as np
import onnxruntime as ort

# From: https://huggingface.co/datasets/wmt14
# which is a part of: https://github.com/huggingface/datasets

# use this for bleu score: https://pytorch.org/text/stable/data_metrics.html

# On brief exploration, looks like a good max number of tokens would be 256.

def beam_decode(decoder_outputs, tok_candidates, scores, pos=0, k=4):
    new_candidates = list(tok_candidates)
    # Returns new tok_candidates and scores
    for i, decoder_output in enumerate(decoder_outputs):
        sorted_logits, sorted_indices = torch.sort(decoder_output[pos], descending=True)
        linears = sorted_logits[:k]
        maxed = F.softmax(linears, dim=0)
        maybe_scores = [scores[i] + maxed[k] for k in range(len(maxed))]
        for k, maybe_score in enumerate(maybe_scores):
            worst_current_i = min(range(len(scores)), key=lambda i: scores[i])
            if maybe_score > scores[worst_current_i]:
                scores[worst_current_i] = maybe_score
                tok_candidates[i][pos] = sorted_indices[k]
                new_candidate = tok_candidates[i]
                new_candidates[worst_current_i] = new_candidate
    return new_candidates, scores

device = "cpu"

# Straight from Vaswani et al
posembeddingmatrix = torch.empty((BATCH_SIZE, bindings['ntokens'], bindings['dmodel']))
for pos in range(len(posembeddingmatrix[0])):
    for i in range(len(posembeddingmatrix[0][0])):
        # Even indices get one thing, odd the other
        if i % 2 == 0:
            posembeddingmatrix[:, pos, i] = math.sin(pos/(10_000**(i/bindings['dmodel'])))
        else:
            posembeddingmatrix[:, pos, i] = math.cos(pos/(10_000**(i/bindings['dmodel'])))
posembeddingmatrix = posembeddingmatrix.to(device)

proto_mask = torch.full((BATCH_SIZE, bindings['ntokens'], bindings['ntokens']), -float("inf"))
proto_mask[:] = torch.triu(proto_mask[0], diagonal=1)
mask = proto_mask
mask = mask.to(device)
zeros_mask = torch.full((BATCH_SIZE, bindings['ntokens'], bindings['ntokens']), 0.).to(device)

if __name__ == '__main__':
    max_len = 0
    german = ""
    english = ""
    other_batch, english_batch = next(load_data(shuffle=False))

    rand_row = random.randrange(len(other_batch))

    english_data = F.one_hot(english_batch.to(torch.int64)).float()

    # (Batch, Seq length)
    chopped = other_batch[:, :-1]
    to_cat = torch.full((BATCH_SIZE, 1), bos_token).to(device)
    other_data = torch.cat((to_cat, chopped), -1).to(torch.int64)
    other_data = F.one_hot(other_data.to(torch.int64)).float()

    example = (other_batch[rand_row], english_batch[rand_row])

    print("Other text:")
    print(get_str_from_ids(example[0]))
    print()
    print("English text:")
    print(get_str_from_ids(example[1]))

    print("Exporting model...")
    agrippa.export("model", "transformer.onnx", index="transformer.agr", bindings=bindings, reinit=False)
    print("Exported")

    torch_model = agrippa.onnx_to_torch("transformer.onnx")
    torch_model = torch_model.to(device)

    ort_sess = ort.InferenceSession('transformer.onnx', providers=['CPUExecutionProvider'])
    outputs = torch_model(other_data[rand_row], english_data[rand_row], mask[rand_row], zeros_mask[rand_row], posembeddingmatrix[rand_row])

    topk = torch.topk(outputs, k=1, dim=1)[1]  # gets indices of top 2 in tensor of shape (seq length, 2)
    tops = topk[:, 0].flatten()

    print()
    print("Top choice of translated output:")

    print(get_str_from_ids(tops))

    # Now doing sampling
    print()
    print("Sampling:")

    k = 1
    presence_penalty = 1
    candidates = [torch.tensor([bos_token for _ in range(SEQ_LENGTH)]) for _ in range(k)]
    scores = [0 for _ in range(k)]
    for i in range(SEQ_LENGTH):
        data_gen_cands = [torch.cat((torch.tensor([bos_token]), cand[:-1]), -1) for cand in candidates]
        data_cands = [F.one_hot(data_gen, num_classes=50257).float() for data_gen in data_gen_cands]

        outputs = []
        for j, data in enumerate(data_cands):
            current = ort_sess.run(None, {'decoder_tokens': data.cpu().detach().numpy(),
                                            'encoder_tokens': english_data[rand_row].cpu().detach().numpy(),
                                            'decoder_mask': mask[rand_row].cpu().detach().numpy(),
                                            'encoder_mask': zeros_mask[rand_row].cpu().detach().numpy(),
                                            'posembedmatrix': posembeddingmatrix[rand_row].cpu().detach().numpy()})
            current = torch.from_numpy(current[0])
            
            # Apply presence penalty
            for id in candidates[j][:i]:
                current[i][id] /= presence_penalty
            
            outputs.append(current)

        candidates, scores = beam_decode(outputs, candidates, scores, pos=i, k=k)
        if i % 10 == 0:
            print(f"At i={i}")
            best_generation = candidates[max(range(len(scores)), key=lambda i: scores[i])]
            print(get_str_from_ids(best_generation))

        all_end = True
        for cand in candidates:
            if cand[i] != bos_token:
                all_end = False
                break
        if all_end:
            break
        
    print("END")
    best_generation = candidates[max(range(len(scores)), key=lambda i: scores[i])]
    print(get_str_from_ids(best_generation))

