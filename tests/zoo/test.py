import sys
sys.path.append('../../src')  # where the agrippa source code is stored
import agrippa
import torch
import math

bindings = {
    'ntokens': 3,
    'nvocab': 5,
    'dmodel': 512,  # 1024 in Transformer Big, 512 in base
    'dffnhidden': 256,  # 4096 in Transformer Big, 2048 in base
    'dvalues': 8,  # 64 in Transformer Big, same in base
    'dqueries': 9,  # 64 in Transformer Big, same in base
    'dkeys': 12,  # 64 in Transformer Big, same in base
    'nheads': 2,  # 16 in Transformer Big, 8 in base
    'nlayers': 3,  # 6 in Transformer Big, same in base
}

constant_bindings = {
    'ln_eps': 1e-5,
    'scale': math.sqrt(bindings['dkeys']),
    'embed_scale': math.sqrt(bindings['dmodel'])
}

bindings.update(constant_bindings)

model_dir = 'transformer'
onnx_path = "testing.onnx"
agrippa.export(model_dir, onnx_path, index="transformer.agr", bindings=bindings)

torch_model = agrippa.onnx_to_torch(onnx_path)

"""

<import from="decoder_tokens" dim="[var(ntokens), var(nvocab)]" />
<import from="encoder_tokens" dim="[var(ntokens), var(nvocab)]" />

<import from="decoder_mask" dim="[var(ntokens), var(ntokens)]" />
<import from="encoder_mask" dim="[var(ntokens), var(ntokens)]" />

<import from="posembedmatrix" dim="[var(ntokens), var(dmodel)]" />

"""

# Test inputs

