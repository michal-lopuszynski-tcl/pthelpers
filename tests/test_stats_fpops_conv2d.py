import pytest
import torch

import pthelpers as pth


class WrapConv2d(torch.nn.Module):
    def __init__(self, *, in_channels, out_channels, kernel_size, padding, bias):
        super().__init__()
        self.op = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, x):
        return self.op(x)


CONV2D_TEST_PARAMS = [
    {
        "expected_fpops": 42549248.0,
        "input_shape": (1, 3, 224, 224),
        "op_params": {
            "in_channels": 3,
            "out_channels": 16,
            "kernel_size": 3,
            "padding": "same",
            "bias": False,
        },
    },
    {
        "expected_fpops": 41792832.0,
        "input_shape": (1, 3, 224, 224),
        "op_params": {
            "in_channels": 3,
            "out_channels": 16,
            "kernel_size": 3,
            "padding": "valid",
            "bias": False,
        },
    },
    {
        "expected_fpops": 119619584.0,
        "input_shape": (1, 3, 224, 224),
        "op_params": {
            "in_channels": 3,
            "out_channels": 16,
            "kernel_size": 5,
            "padding": "same",
            "bias": False,
        },
    },
    {
        "expected_fpops": 43352064.0,
        "input_shape": (1, 3, 224, 224),
        "op_params": {
            "in_channels": 3,
            "out_channels": 16,
            "kernel_size": 3,
            "padding": "same",
            "bias": True,
        },
    },
    {
        "expected_fpops": 43352064.0,
        "input_shape": (3, 3, 224, 224),
        "op_params": {
            "in_channels": 3,
            "out_channels": 16,
            "kernel_size": 3,
            "padding": "same",
            "bias": True,
        },
    },
    {
        "expected_fpops": 120422400.0,
        "input_shape": (1, 3, 224, 224),
        "op_params": {
            "in_channels": 3,
            "out_channels": 16,
            "kernel_size": 5,
            "padding": "same",
            "bias": True,
        },
    },
    {
        "expected_fpops": 222298112.0,
        "input_shape": (1, 3, 512, 512),
        "op_params": {
            "in_channels": 3,
            "out_channels": 16,
            "kernel_size": 3,
            "padding": "same",
            "bias": False,
        },
    },
    {
        "expected_fpops": 220564800.0,
        "input_shape": (1, 3, 512, 512),
        "op_params": {
            "in_channels": 3,
            "out_channels": 16,
            "kernel_size": 3,
            "padding": "valid",
            "bias": False,
        },
    },
    {
        "expected_fpops": 624951296.0,
        "input_shape": (1, 3, 512, 512),
        "op_params": {
            "in_channels": 3,
            "out_channels": 16,
            "kernel_size": 5,
            "padding": "same",
            "bias": False,
        },
    },
    {
        "expected_fpops": 226492416.0,
        "input_shape": (1, 3, 512, 512),
        "op_params": {
            "in_channels": 3,
            "out_channels": 16,
            "kernel_size": 3,
            "padding": "same",
            "bias": True,
        },
    },
    {
        "expected_fpops": 224726400.0,
        "input_shape": (1, 3, 512, 512),
        "op_params": {
            "in_channels": 3,
            "out_channels": 16,
            "kernel_size": 3,
            "padding": "valid",
            "bias": True,
        },
    },
    {
        "expected_fpops": 224726400.0,
        "input_shape": (3, 3, 512, 512),
        "op_params": {
            "in_channels": 3,
            "out_channels": 16,
            "kernel_size": 3,
            "padding": "valid",
            "bias": True,
        },
    },
    {
        "expected_fpops": 629145600.0,
        "input_shape": (1, 3, 512, 512),
        "op_params": {
            "in_channels": 3,
            "out_channels": 16,
            "kernel_size": 5,
            "padding": "same",
            "bias": True,
        },
    },
    {
        "expected_fpops": 28700672.0,
        "input_shape": (1, 8, 224, 224),
        "op_params": {
            "in_channels": 8,
            "out_channels": 4,
            "kernel_size": 3,
            "padding": "same",
            "bias": False,
        },
    },
    {
        "expected_fpops": 28190448.0,
        "input_shape": (1, 8, 224, 224),
        "op_params": {
            "in_channels": 8,
            "out_channels": 4,
            "kernel_size": 3,
            "padding": "valid",
            "bias": False,
        },
    },
    {
        "expected_fpops": 80080896.0,
        "input_shape": (1, 8, 224, 224),
        "op_params": {
            "in_channels": 8,
            "out_channels": 4,
            "kernel_size": 5,
            "padding": "same",
            "bias": False,
        },
    },
    {
        "expected_fpops": 28901376.0,
        "input_shape": (1, 8, 224, 224),
        "op_params": {
            "in_channels": 8,
            "out_channels": 4,
            "kernel_size": 3,
            "padding": "same",
            "bias": True,
        },
    },
    {
        "expected_fpops": 28387584.0,
        "input_shape": (1, 8, 224, 224),
        "op_params": {
            "in_channels": 8,
            "out_channels": 4,
            "kernel_size": 3,
            "padding": "valid",
            "bias": True,
        },
    },
    {
        "expected_fpops": 80281600.0,
        "input_shape": (1, 8, 224, 224),
        "op_params": {
            "in_channels": 8,
            "out_channels": 4,
            "kernel_size": 5,
            "padding": "same",
            "bias": True,
        },
    },
]


@pytest.mark.parametrize("test_params", CONV2D_TEST_PARAMS)
def test_calc_fpops_conv2d(test_params):
    expected_fpops = test_params["expected_fpops"]
    input_shape = test_params["input_shape"]
    op_params = test_params["op_params"]

    m = WrapConv2d(**op_params)
    fpops = pth.stats.calc_fpops(m, input_shapes=(input_shape,), unit="flops")
    assert fpops == expected_fpops
