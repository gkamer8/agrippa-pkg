This folder contains a processed dataset of the Wikipedia English corpus.

The source of this dataset is here: https://huggingface.co/datasets/lucadiliello/english_wikipedia/tree/main/data

So far, only a subset of the corpus has been processed. In particular, the following .parquet files (located in 'wiki-raw'), have been processed:
- train5
- train7
- train8
- train9
- train10
- train12
- train13
- train14
- train16
- train19
- train20

Leaving:
- train1
- train2
- train3
- train4
- train6
- train11
- train15
- train17
- train18
- train21
- train22

left to do. Thus about half of the corpus is actually processed so far.

With 12,740,633 chunks of sequences sized 64 tokens, the total training set consists of 815m tokens.