import chunk
import pandas as pd
import os
from transformers import GPT2Tokenizer
import pickle
import numpy as np

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

chunk_folder = "chunks"

# Chop up tokens into sequences
def sequify(tokens, seq_length):
    # Ignore the final thing for now, so everything is truly of seq length
    sequences = []
    start_i = 0
    for i in range(start_i, len(tokens), seq_length):
        right_side = min(i+seq_length, len(tokens))
        if right_side - i != seq_length:
            continue
        to_add = tokens[i:right_side]
        sequences.append(to_add)
    return sequences
        

# Chop up data into sequences
def chunkify(chunk_size=64):
    fldr = "wiki-raw"
    outfolder = chunk_folder
    if outfolder not in os.listdir():
        os.mkdir(os.path.join(outfolder))
    # Go through parquet files in wiki-raw
    parquets = os.listdir(fldr)
    chunk_num = 0
    for par in parquets:
        print(f"At {par} in {parquets}")
        df = pd.read_parquet(os.path.join(fldr, par))
        # Go through each row
        for i in range(len(df['maintext'])):
            txt = df['maintext'][i]
            tokens = tokenizer(txt)['input_ids']
            seqs = sequify(tokens, chunk_size)
            for seq in seqs:
                assert(len(seq) == chunk_size)
                with open(os.path.join(outfolder, f"{chunk_num}.pkl"), 'wb') as fhand:
                    pickle.dump(seq, fhand)
                chunk_num += 1
            if i % 1000 == 0:
                print(f"{i+1}/{len(df['maintext'])}")
        print(f"Done with {par}")

# Turn chunks into batches
# Batches include examples randomly chosen from dataset
def batchify(batch_size=200, outfolder="batches"):
    chunks = os.listdir(chunk_folder)
    print("Got chunks")
    max_chunk = max(chunks, key=lambda x: int(x.split(".")[0]))
    max_chunk = int(max_chunk.split(".")[0])
    print(f"Got max chunk ({max_chunk})")
    chunk_nums = [i for i in range(max_chunk+1)]
    np.random.shuffle(chunk_nums)
    print("Shuffled chunks.")

    if outfolder not in os.listdir():
        print("Making batches folder")
        os.mkdir(os.path.join(outfolder))
    
    batch_num = 0
    for batch in range(0, max_chunk+1, batch_size):
        right_side = min(batch+batch_size, max_chunk+1)
        if right_side - batch != batch_size:
            print("Bad batch size")
            continue
        batch_data = []
        for k in range(batch, right_side):
            with open(os.path.join("chunks", f'{chunk_nums[k]}.pkl'), 'rb') as fhand:
                x = pickle.load(fhand)
                batch_data.append(x)
        with open(os.path.join(outfolder, f"{batch_num}.pkl"), 'wb') as fhand:
            if batch == 0 or batch_num == 1:
                print("Sample batch:")
                print(batch_data)
                print(len(batch_data))
            pickle.dump(batch_data, fhand)
        if batch_num % 100 == 0:
            print(f"At batch {batch_num+1}")
        batch_num += 1

# chunkify()
batchify()