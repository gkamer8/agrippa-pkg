__all__ = [
    'OnnxReciprocal',
]

import torch
from torch import nn

from agrippa.onnx2torch.node_converters.registry import add_converter
from agrippa.onnx2torch.onnx_graph import OnnxGraph
from agrippa.onnx2torch.onnx_node import OnnxNode
from agrippa.onnx2torch.utils.common import OnnxToTorchModule
from agrippa.onnx2torch.utils.common import OperationConverterResult
from agrippa.onnx2torch.utils.common import onnx_mapping_from_node


class OnnxReciprocal(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    def forward(self, x):  # pylint: disable=missing-function-docstring
        return torch.reciprocal(x)


@add_converter(operation_type='Reciprocal', version=1)
@add_converter(operation_type='Reciprocal', version=6)
@add_converter(operation_type='Reciprocal', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxReciprocal(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
