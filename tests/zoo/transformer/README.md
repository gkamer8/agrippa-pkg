# Transformer

This is the original Transformer architecture as seen in Vaswani et al. 2017. This implementation accepts five inputs:

```
<import from="decoder_tokens" dim="[var(ntokens), var(nvocab)]" />
<import from="encoder_tokens" dim="[var(ntokens), var(nvocab)]" />

<import from="decoder_mask" dim="[var(ntokens), var(ntokens)]" />
<import from="encoder_mask" dim="[var(ntokens), var(ntokens)]" />

<import from="posembedmatrix" dim="[var(ntokens), var(dmodel)]" />
```

And outputs a single vector, which represents the logits:

```
<export from="end_linear" dim="[var(ntokens), var(nvocab)]" />
```

Finally, here are some example variable bindings:

```
# Base model from vaswani et al
bindings = {
    'ntokens': SEQ_LENGTH,
    'nvocab': 50257,
    'dmodel': 512,  # 1024 in Transformer Big, 512 in base
    'dffnhidden': 2048,  # 4096 in Transformer Big, 2048 in base
    'dvalues': 64,  # 64 in Transformer Big, same in base
    'dqueries': 64,  # 64 in Transformer Big, same in base
    'dkeys': 64,  # 64 in Transformer Big, same in base
    'nheads': 8,  # 16 in Transformer Big, 8 in base
    'nlayers': 6,  # 6 in Transformer Big, same in base
    'scale': math.sqrt(64),  # scaled division inside attention
    'embed_scale': math.sqrt(512)  # multiplied by input embedding matrix
}
```
