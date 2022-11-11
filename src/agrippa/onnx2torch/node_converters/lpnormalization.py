import torch
from torch import nn

from agrippa.onnx2torch.node_converters.registry import add_converter
from agrippa.onnx2torch.onnx_graph import OnnxGraph
from agrippa.onnx2torch.onnx_node import OnnxNode
from agrippa.onnx2torch.utils.common import OnnxToTorchModule
from agrippa.onnx2torch.utils.common import OperationConverterResult
from agrippa.onnx2torch.utils.common import onnx_mapping_from_node


class OnnxLpNormalization(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(self, axis, p):
        super().__init__()
        self.axis = axis
        if not p:
            self.p = -1
        else:
            self.p = p

    def forward(self, x):  # pylint: disable=missing-function-docstring
        # An epsilon value is added for stability; we use the default pytorch epsilon
        return torch.nn.functional.normalize(x, p=self.p, dim=self.axis)


@add_converter(operation_type='LpNormalization', version=1)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxLpNormalization(
            axis=node.attributes['axis'],
            p=node.attributes['p']
        ),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
