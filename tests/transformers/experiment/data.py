from lib2to3.pgen2 import token
from transformers import GPT2Tokenizer
import os
import pickle
import numpy as np
import json

"""

Code to process the dataset

"""

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def get_ids_to_toks():
    return tokenizer.convert_ids_to_tokens

def get_toks_to_str():
    return tokenizer.convert_tokens_to_string

def get_n_batches(path):
    with open(path, 'r') as fhand:
        x = json.load(fhand)
        return x['nbatches']

"""
Looks at a text file, breaks it up into chunks of size chunk_size, outputs
    chunks as i.pkl, with i starting at start_i, into outfolder

Chunks break up on a space
"""
def chunkify(folder, outfolder, chunk_size, start_i=0, nested_folder='raw-books'):
    tokenizer.save_vocabulary('vocab')
    if outfolder not in os.listdir(folder):
        os.mkdir(os.path.join(folder, outfolder))
    books = os.listdir(os.path.join(folder, nested_folder))
    for book in books:
        print(f"Reading {book}")
        with open(os.path.join(folder, nested_folder, book)) as fhand:
            my_str = fhand.read()
            tokens = tokenizer(my_str)['input_ids']
            for i in range(start_i, len(tokens), chunk_size):
                right_side = min(i+chunk_size, len(tokens))
                to_add = tokens[i:right_side]
                new_name = f"{book}_{i}.pkl"
                with open(os.path.join(folder, outfolder, new_name), 'wb') as small_f:
                    pickle.dump(to_add, small_f)

def batchify(folder, outfolder, batch_size, chunk_folder):
    cf = os.path.join(folder, chunk_folder)
    examples = os.listdir(cf)

    if outfolder not in os.listdir(folder):
        os.mkdir(os.path.join(folder, outfolder))

    nbatches = 0
    max_chunk_size = 0
    np.random.shuffle(examples)
    for i in range(0, len(examples), batch_size):
        batch = examples[i:min(i+batch_size, len(examples))]
        batch_data = []
        for filename in batch:
            example = os.path.join(folder, chunk_folder, filename)
            with open(example, 'rb') as fhand:
                x = pickle.load(fhand)
                max_chunk_size = max(max_chunk_size, len(x))
                batch_data.append(x)
        newname = f"{i//batch_size}.pkl"
        with open(os.path.join(folder, outfolder, newname), 'wb') as fhand:
            pickle.dump(batch_data, fhand)
        nbatches += 1

    # Create a meta file
    meta_file = os.path.join(folder, outfolder, 'meta.json')
    with open(meta_file, 'w') as fhand:
        json_obj = {
            'nbatches': nbatches,
            'chunk_folder': chunk_folder,
            'batch_size': batch_size,
            'max_chunk_size': max_chunk_size
        }
        json.dump(json_obj, fhand)


