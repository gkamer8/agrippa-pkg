from torchtext.data.metrics import bleu_score
from preprocess import load_data
import onnxruntime as ort
import torch
from data_constants import bos_token, SEQ_LENGTH, BATCH_SIZE
from preprocess import get_str_from_ids
import sys
sys.path.append('../../../src')  # where the agrippa source code is stored
import agrippa
import torch.nn.functional as F
from train import bindings
from test import mask, posembeddingmatrix, zeros_mask, beam_decode, device


if __name__ == '__main__':

    print("Exporting model...")
    # agrippa.export("model", "transformer.onnx", index="transformer.agr", bindings=bindings, reinit=False)
    print("Exported")

    ort_sess = ort.InferenceSession('transformer.onnx', providers=['CPUExecutionProvider'])

    for other_batch, english_batch in load_data(split="validation"):
        for row in range(len(other_batch)):
            english_data = F.one_hot(english_batch.to(torch.int64)).float()

            # (Batch, Seq length)
            chopped = other_batch[:, :-1]
            to_cat = torch.full((BATCH_SIZE, 1), bos_token).to(device)
            other_data = torch.cat((to_cat, chopped), -1).to(torch.int64)
            other_data = F.one_hot(other_data.to(torch.int64)).float()

            example = (other_batch[row], english_batch[row])
            k = 4
            presence_penalty = 2
            candidates = [torch.tensor([bos_token for _ in range(SEQ_LENGTH)]) for _ in range(k)]
            scores = [0 for _ in range(k)]
            for i in range(SEQ_LENGTH):
                data_gen_cands = [torch.cat((torch.tensor([bos_token]), cand[:-1]), -1) for cand in candidates]
                data_cands = [F.one_hot(data_gen, num_classes=50257).float() for data_gen in data_gen_cands]

                outputs = []
                for j, data in enumerate(data_cands):
                    current = ort_sess.run(None, {'decoder_tokens': data.cpu().detach().numpy(),
                                                    'encoder_tokens': english_data[row].cpu().detach().numpy(),
                                                    'decoder_mask': mask[row].cpu().detach().numpy(),
                                                    'encoder_mask': zeros_mask[row].cpu().detach().numpy(),
                                                    'posembedmatrix': posembeddingmatrix[row].cpu().detach().numpy()})
                    current = torch.from_numpy(current[0])
                    
                    # Apply presence penalty
                    for id in candidates[j][:i]:
                        current[i][id] /= presence_penalty
                    
                    outputs.append(current)

                candidates, scores = beam_decode(outputs, candidates, scores, pos=i, k=k)
                if i % 5 == 0:
                    print(f"At i={i}")

                all_end = True
                for cand in candidates:
                    if cand[i] != bos_token:
                        all_end = False
                        break
                if all_end:
                    break
                
            print("END")
            best_generation = candidates[max(range(len(scores)), key=lambda i: scores[i])]
            print("Translation:")
            print(get_str_from_ids(best_generation))

            print("English:")
            print(get_str_from_ids(english_batch[row]))

            exit(0)
        