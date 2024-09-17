# Implements symmetric quantization.
import torch
from enum import Enum


class QuantizationType(Enum):
    PER_TENSOR = "PER_TENSOR"
    PER_INPUT_CHANNEL = "PER_INPUT_CHANNEL"


class QuantizationMethod(Enum):
    ABSMAX = "ABSMAX"


def quantize_weight_symmetric(
    w: torch.Tensor,
    quant_type: QuantizationType,
    quant_method: QuantizationMethod,
    n_bits: int,
):
    """
    w of shape (out_channels, in_channels)
    """
    qmax = 2 ** (n_bits - 1) - 1
    if quant_method is QuantizationMethod.ABSMAX:
        wabs = w.abs()
        if quant_type is QuantizationType.PER_TENSOR:
            scale = wabs.max()
        elif quant_type is QuantizationType.PER_INPUT_CHANNEL:
            scale = wabs.max(dim=-1, keepdim=True)
    scale = scale.clamp(min=1e-8) / qmax
    w = torch.round(w / scale).int()
    w = w.clamp(-qmax, qmax)
    return w, scale


class W8A8Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        weight_scale: torch.Tensor,
        input_scale: torch.Tensor,
        output_scale: torch.Tensor,
    ):
        super(W8A8Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.zeros(out_features, in_features).int(), requires_grad=False
        )
        self.weight_scale = torch.nn.Parameter(weight_scale)
        self.input_scale = torch.nn.Parameter(input_scale)
        self.output_scale = torch.nn.Parameter(output_scale)
        if bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(out_features).int(), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(
        cls,
        linear: torch.nn.Linear,
        quant_type: QuantizationType,
        quant_method: QuantizationMethod,
        n_bits: int,
        input_scale: torch.Tensor,
        output_scale: torch.Tensor,
    ):
        w, scale = quantize_weight_symmetric(
            linear.weight, quant_type, quant_method, n_bits
        )
        new_linear = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            scale,
            input_scale,
            output_scale,
        )
        new_linear.weight.data = w
        if linear.bias is not None:
            new_linear.bias.data = linear.bias.int()
        return new_linear

    def forward(self, x):
        # quantize the input
        x = torch.round(x / self.input_scale).int()
        # dequanrize it
        x = x * self.input_scale

        # full mm
        w = self.weight.float() * self.weight_scale
        ret = x @ w.t()
        if self.bias is not None:
            ret += self.bias.float()

        # quantize the output
        ret = torch.round(ret / self.output_scale).int()

        return ret
