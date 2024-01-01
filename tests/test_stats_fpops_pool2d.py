import pytest
import torch

import pthelpers as pth


class WrapPool2d(torch.nn.Module):
    def __init__(self, *, type, kernel_size, padding, stride):
        super().__init__()
        if type == "avg":
            self.op = torch.nn.AvgPool2d(
                kernel_size=kernel_size, padding=padding, stride=stride
            )
        elif type == "max":
            self.op = torch.nn.MaxPool2d(
                kernel_size=kernel_size, padding=padding, stride=stride
            )

    def forward(self, x):
        return self.op(x)


POOL2D_TEST_PARAMS = [
    {
        "expected_fpops": 1330668.0,
        "expected_output_shape": (1, 3, 222, 222),
        "input_shape": (1, 3, 224, 224),
        "op_params": {"type": "avg", "kernel_size": 3, "stride": 1, "padding": 0},
    },
    {
        "expected_fpops": 1330668.0,
        "expected_output_shape": (4, 3, 222, 222),
        "input_shape": (4, 3, 224, 224),
        "op_params": {"type": "avg", "kernel_size": 3, "stride": 1, "padding": 0},
    },
    {
        "expected_fpops": 1742400.0,
        "expected_output_shape": (1, 16, 110, 110),
        "input_shape": (1, 16, 112, 112),
        "op_params": {"type": "avg", "kernel_size": 3, "stride": 1, "padding": 0},
    },
    {
        "expected_fpops": 3630000.0,
        "expected_output_shape": (1, 3, 220, 220),
        "input_shape": (1, 3, 224, 224),
        "op_params": {"type": "avg", "kernel_size": 5, "stride": 1, "padding": 0},
    },
    {
        "expected_fpops": 2851200.0,
        "expected_output_shape": (1, 16, 108, 110),
        "input_shape": (1, 16, 112, 112),
        "op_params": {"type": "avg", "kernel_size": (5, 3), "stride": 1, "padding": 0},
    },
    {
        "expected_fpops": 2851200.0,
        "expected_output_shape": (1, 16, 110, 108),
        "input_shape": (1, 16, 112, 112),
        "op_params": {"type": "avg", "kernel_size": (3, 5), "stride": 1, "padding": 0},
    },
    {
        "expected_fpops": 2851200.0,
        "expected_output_shape": (1, 16, 108, 110),
        "input_shape": (1, 16, 112, 112),
        "op_params": {"type": "max", "kernel_size": (5, 3), "stride": 1, "padding": 0},
    },
    {
        "expected_fpops": 2851200.0,
        "expected_output_shape": (1, 16, 110, 108),
        "input_shape": (1, 16, 112, 112),
        "op_params": {"type": "max", "kernel_size": (3, 5), "stride": 1, "padding": 0},
    },
]


@pytest.mark.parametrize("test_params", POOL2D_TEST_PARAMS)
def test_calc_fpops_pool2d(test_params):
    expected_fpops = test_params["expected_fpops"]
    expected_output_shape = test_params["expected_output_shape"]
    op_params = test_params["op_params"]
    input_shape = test_params["input_shape"]

    m = WrapPool2d(**op_params)

    sample_inputs = pth.make_rand_tensors((input_shape,), torch.device("cpu"))
    fpops = pth.stats.calc_fpops(m, input_shapes=(input_shape,), unit="flops")
    output = m(sample_inputs[0])
    output_shape = tuple(output.shape)
    assert output_shape == expected_output_shape
    assert fpops == expected_fpops
