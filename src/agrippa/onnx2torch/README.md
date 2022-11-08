# onnx2torch

The original onnx2torch package can be found [here](https://github.com/ENOT-AutoDL/agrippa.onnx2torch).

It does a lot of the core work for us, but its precise functionality needed to be modified in order to support, inter alia, training. In particular, the original package registers initializations as buffers; however, we need to register initializations as parameters so that they can be trained. Furthermore, because users specify parameters in the markup language, we can tell which initializations are parameters and which are constants. Thus we can add additional control over the conversion process.

The original package is distributed under the Apache license, which requires that the license agreement perpertuate in derivative works. I've done that by adding a license file. I also specify the modifications made. Further modifications should be added to the license. The version of onnx2torch that was imported before modifications were made was version `1.5.3`.

It is included here as a subdirectory of agrippa. All of the imports have been changed to include agrippa.onnx2torch rather than onnx2torch via a grep/sed, which was the easiest way I thought to do it.
