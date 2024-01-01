import pytest
import torch

import pthelpers as pth


class WrapDwiseConv2d(torch.nn.Module):
    def __init__(self, *, in_channels, kernel_size, padding, bias):
        super().__init__()
        self.op = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, x):
        return self.op(x)


CONV2D_DWISE_TEST_PARAMS = [
    {
        "expected_fpops": 52224.0,
        "input_shape": (1, 3, 32, 32),
        "op_params": {
            "in_channels": 3,
            "kernel_size": 3,
            "padding": "same",
            "bias": False,
        },
        "expected_output_shape": (1, 3, 32, 32),
    },
    {
        "expected_fpops": 45900.0,
        "input_shape": (1, 3, 32, 32),
        "op_params": {
            "in_channels": 3,
            "kernel_size": 3,
            "padding": "valid",
            "bias": False,
        },
        "expected_output_shape": (1, 3, 30, 30),
    },
    {
        "expected_fpops": 150528.0,
        "input_shape": (1, 3, 32, 32),
        "op_params": {
            "in_channels": 3,
            "kernel_size": 5,
            "padding": "same",
            "bias": False,
        },
        "expected_output_shape": (1, 3, 32, 32),
    },
    {
        "expected_fpops": 55296.0,
        "input_shape": (1, 3, 32, 32),
        "op_params": {
            "in_channels": 3,
            "kernel_size": 3,
            "padding": "same",
            "bias": True,
        },
        "expected_output_shape": (1, 3, 32, 32),
    },
    {
        "expected_fpops": 2661336.0,
        "input_shape": (1, 3, 224, 224),
        "op_params": {
            "in_channels": 3,
            "kernel_size": 3,
            "padding": "valid",
            "bias": True,
        },
        "expected_output_shape": (1, 3, 222, 222),
    },
    {
        "expected_fpops": 153600.0,
        "input_shape": (1, 3, 32, 32),
        "op_params": {
            "in_channels": 3,
            "kernel_size": 5,
            "padding": "same",
            "bias": True,
        },
        "expected_output_shape": (1, 3, 32, 32),
    },
    {
        "expected_fpops": 1392640.0,
        "input_shape": (1, 5, 128, 128),
        "op_params": {
            "in_channels": 5,
            "kernel_size": 3,
            "padding": "same",
            "bias": False,
        },
        "expected_output_shape": (1, 5, 128, 128),
    },
    {
        "expected_fpops": 1349460.0,
        "input_shape": (1, 5, 128, 128),
        "op_params": {
            "in_channels": 5,
            "kernel_size": 3,
            "padding": "valid",
            "bias": False,
        },
        "expected_output_shape": (1, 5, 126, 126),
    },
    {
        "expected_fpops": 4014080.0,
        "input_shape": (1, 5, 128, 128),
        "op_params": {
            "in_channels": 5,
            "kernel_size": 5,
            "padding": "same",
            "bias": False,
        },
        "expected_output_shape": (1, 5, 128, 128),
    },
    {
        "expected_fpops": 1474560.0,
        "input_shape": (1, 5, 128, 128),
        "op_params": {
            "in_channels": 5,
            "kernel_size": 3,
            "padding": "same",
            "bias": True,
        },
        "expected_output_shape": (1, 5, 128, 128),
    },
]


@pytest.mark.parametrize("test_params", CONV2D_DWISE_TEST_PARAMS)
def test_calc_fpops_conv2d_dwise(test_params):
    expected_fpops = test_params["expected_fpops"]
    expected_output_shape = test_params["expected_output_shape"]
    input_shape = test_params["input_shape"]
    op_params = test_params["op_params"]
    m = WrapDwiseConv2d(**op_params)
    fpops = pth.stats.calc_fpops(m, input_shapes=(input_shape,), unit="flops")
    sample_inputs = pth.make_rand_tensors((input_shape,), torch.device("cpu"))
    output = m(sample_inputs[0])
    output_shape = tuple(output.shape)
    assert output_shape == expected_output_shape
    assert fpops == expected_fpops
