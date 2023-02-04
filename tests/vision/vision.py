import sys

sys.path.append('../../src')  # where the agrippa source code is stored
import agrippa

import torch
import numpy as np
import onnxruntime as ort

x = np.array(
    [
        [
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],  # (1, 1, 5, 5) input tensor
                [5.0, 6.0, 7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0, 18.0, 19.0],
                [20.0, 21.0, 22.0, 23.0, 24.0],
            ]
        ]
    ]
).astype(np.float32)

bindings = {
    'C': 1,  # number of channels in image
    'H': 5,  # height of image
    'W': 5,  # width of image
    'batch': 1,  # batch size
    'kH': 3,  # kernel height
    'kW': 3,  # kernel width
    'M': 1,  # number of feature maps
    'oH': 5,  # output height
    'oW': 5  # output width
}

proj = "model"
outfile = "conv_net.onnx"

agrippa.export(proj, outfile, bindings=bindings)

torch_model = agrippa.onnx_to_torch(outfile)

ort_sess = ort.InferenceSession(outfile, providers=['CPUExecutionProvider'])
outputs = ort_sess.run(None, {'image': x})

print(outputs)
print(torch_model(torch.tensor(x)))
