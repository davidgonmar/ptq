# Implements symmetric quantization.
import torch
from enum import Enum


class QuantizationType(Enum):
    PER_TENSOR = "PER_TENSOR"
    PER_INPUT_CHANNEL = "PER_INPUT_CHANNEL"


class QuantizationMethod(Enum):
    ABSMAX = "ABSMAX"


class QuantizationTime(Enum):
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"


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


@torch.no_grad()
def online_quantize_activation_per_tensor_absmax(t, n_bits=8):
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


class FakeW8A8Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        weight_scale: torch.Tensor,
        act_quant_time: QuantizationTime,
        input_scale: torch.Tensor = None,
        output_scale: torch.Tensor = None,
    ):
        super(FakeW8A8Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.zeros(out_features, in_features), requires_grad=False
        )
        if act_quant_time is QuantizationTime.ONLINE:
            self.register_buffer("input_scale", None)
            self.register_buffer("output_scale", None)
        else:
            assert (
                input_scale is not None and output_scale is not None
            ), "input and output scale must be provided for offline quantization"
            self.input_scale = torch.nn.Parameter(input_scale)
            self.output_scale = torch.nn.Parameter(output_scale)
        self.weight_scale = torch.nn.Parameter(weight_scale)
        self.act_quant_time = act_quant_time
        if bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(out_features), requires_grad=False
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
        act_quant_time: QuantizationTime,
        input_scale: torch.Tensor = None,
        output_scale: torch.Tensor = None,
    ):
        w, w_scale = quantize_weight_symmetric(
            linear.weight, quant_type, quant_method, n_bits
        )
        new_linear = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            w_scale,
            act_quant_time,
            input_scale,
            output_scale,
        )
        new_linear.weight.data = w.float()  # fake quantized weight
        if linear.bias is not None:
            new_linear.bias.data = linear.bias
        return new_linear

    def forward(self, x):
        # quantize the input
        if self.act_quant_time is QuantizationTime.ONLINE:
            x = online_quantize_activation_per_tensor_absmax(x)
        else:
            x = torch.round(x / self.input_scale) * self.input_scale
        # x = online_quantize_activation_per_tensor_absmax(x)
        # full mm
        ret = (x @ self.weight.t()) * self.weight_scale
        if self.bias is not None:
            ret += self.bias
        # quantize the output
        # ret = online_quantize_activation_per_tensor_absmax(ret)
        if self.act_quant_time is QuantizationTime.ONLINE:
            ret = online_quantize_activation_per_tensor_absmax(ret)
        else:
            ret = torch.round(ret / self.output_scale) * self.output_scale
        return ret
