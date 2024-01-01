import pytest
import torch

import pthelpers as pth


class WrapPool2dAdaptive(torch.nn.Module):
    def __init__(self, *, type, output_size):
        super().__init__()
        if type == "avg":
            self.op = torch.nn.AdaptiveAvgPool2d(output_size=output_size)
        elif type == "max":
            self.op = torch.nn.AdaptiveMaxPool2d(output_size=output_size)
        else:
            raise ValueError(f"Unkown {type=}")

    def forward(self, x):
        return self.op(x)


POOL2D_ADAPTIVE_TEST_PARAMS = [
    {
        "expected_fpops": 3072.0,
        "expected_output_shape": (1, 3, 4, 4),
        "input_shape": (1, 3, 32, 32),
        "op_params": {"type": "avg", "output_size": (4, 4)},
    },
    {
        "expected_fpops": 3072.0,
        "expected_output_shape": (5, 3, 4, 4),
        "input_shape": (5, 3, 32, 32),
        "op_params": {"type": "avg", "output_size": (4, 4)},
    },
    {
        "expected_fpops": 3072.0,
        "expected_output_shape": (1, 3, 4, 4),
        "input_shape": (1, 3, 32, 32),
        "op_params": {"type": "max", "output_size": (4, 4)},
    },
    {
        "expected_fpops": 192.0,
        "expected_output_shape": (1, 3, 1, 1),
        "input_shape": (1, 3, 8, 8),
        "op_params": {"type": "avg", "output_size": (1, 1)},
    },
    {
        "expected_fpops": 192.0,
        "expected_output_shape": (5, 3, 1, 1),
        "input_shape": (5, 3, 8, 8),
        "op_params": {"type": "avg", "output_size": (1, 1)},
    },
    {
        "expected_fpops": 192.0,
        "expected_output_shape": (1, 3, 1, 1),
        "input_shape": (1, 3, 8, 8),
        "op_params": {"type": "max", "output_size": (1, 1)},
    },
]


@pytest.mark.parametrize("test_params", POOL2D_ADAPTIVE_TEST_PARAMS)
def test_calc_fpops_pool2d_nonadaptive(test_params):
    expected_fpops = test_params["expected_fpops"]
    expected_output_shape = test_params["expected_output_shape"]
    op_params = test_params["op_params"]
    input_shape = test_params["input_shape"]

    m = WrapPool2dAdaptive(**op_params)

    sample_inputs = pth.make_rand_tensors((input_shape,), torch.device("cpu"))
    fpops = pth.stats.calc_fpops(m, input_shapes=(input_shape,), unit="flops")
    output = m(sample_inputs[0])
    output_shape = tuple(output.shape)
    assert output_shape == expected_output_shape
    assert fpops == expected_fpops
