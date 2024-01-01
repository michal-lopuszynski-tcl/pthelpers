import pytest
import torch

import pthelpers as pth


class WrapLinear(torch.nn.Module):
    def __init__(self, *, in_features, out_features, bias):
        super().__init__()
        self.op = torch.nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

    def forward(self, x):
        return self.op(x)


LINEAR_TEST_PARAMS = [
    {
        "expected_fpops": 4096.0,
        "input_shape": (1, 128),
        "op_params": {"in_features": 128, "out_features": 16, "bias": False},
    },
    {
        "expected_fpops": 12288.0,
        "input_shape": (1, 3, 128),
        "op_params": {"in_features": 128, "out_features": 16, "bias": False},
    },
    {
        "expected_fpops": 61440.0,
        "input_shape": (1, 3, 5, 128),
        "op_params": {"in_features": 128, "out_features": 16, "bias": False},
    },
    {
        "expected_fpops": 4112.0,
        "input_shape": (1, 128),
        "op_params": {"in_features": 128, "out_features": 16, "bias": True},
    },
    {
        "expected_fpops": 4112.0,
        "input_shape": (3, 128),
        "op_params": {"in_features": 128, "out_features": 16, "bias": True},
    },
    {
        "expected_fpops": 12336.0,
        "input_shape": (1, 3, 128),
        "op_params": {"in_features": 128, "out_features": 16, "bias": True},
    },
    {
        "expected_fpops": 61680.0,
        "input_shape": (1, 3, 5, 128),
        "op_params": {"in_features": 128, "out_features": 16, "bias": True},
    },
    {
        "expected_fpops": 61440.0,
        "input_shape": (1, 3, 5, 128),
        "op_params": {"in_features": 128, "out_features": 16, "bias": False},
    },
    {
        "expected_fpops": 61680.0,
        "input_shape": (3, 3, 5, 128),
        "op_params": {"in_features": 128, "out_features": 16, "bias": True},
    },
    {
        "expected_fpops": 61440.0,
        "input_shape": (3, 3, 5, 128),
        "op_params": {"in_features": 128, "out_features": 16, "bias": False},
    },
]


@pytest.mark.parametrize("test_params", LINEAR_TEST_PARAMS)
def test_calc_fpops_linear(test_params):
    expected_fpops = test_params["expected_fpops"]
    op_params = test_params["op_params"]
    input_shape = test_params["input_shape"]

    m = WrapLinear(**op_params)
    fpops = pth.stats.calc_fpops(m, input_shapes=(input_shape,), unit="flops")
    _ = m(torch.rand(input_shape))
    assert fpops == expected_fpops
