# Embedding Model

Goal here is to train an encoder model that produces an embedding representing a document.

Works like this:

ENCODER -> DECODER -> trained autoregressively

## Tricks

Document embedding is last embedding outputted from the encoder. That embedding gets passed to decoder. In the implementation here, technically we mask off all the outputs from the decoder except the last one (doc embedding) and add it to the embeddings of the decoder. Since the embedding was the last one in the encoder, it will be the last one in the decoder, so we need to change the posembeddings and decoder mask appropriately. The decoder is getting input that is essentially right shifted (to put BOS) and then with the last embedding replaced by the doc embedding (so it's missing two tokens that are in the output). In order to avoid problems, the last embedding in the decoder gets masked off (basically inverse of the mask after the encoder output). The encoder is also missing two tokens, since the last token is replaced by a special summarize token.

## Running

Set the data_constants appropriately and then run the train script.