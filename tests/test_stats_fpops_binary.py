import pytest
import torch

import pthelpers as pth


class WrapBinary(torch.nn.Module):
    def __init__(self, *, op):
        super().__init__()
        self.op = op

    def forward(self, x1, x2):
        if self.op == "add":
            return x1 + x2
        elif self.op == "mul":
            return x1 * x2
        else:
            raise ValueError(f"Unknown op {self.op=}")


BINARY_TEST_PARAMS = [
    {
        "expected_fpops": 150528.0,
        "expected_output_shape": (1, 3, 224, 224),
        "input_shapes": ((1, 3, 224, 224), (1, 3, 224, 224)),
        "op_params": {"op": "add"},
    },
    {
        "expected_fpops": 150528.0,
        "expected_output_shape": (1, 3, 224, 224),
        "input_shapes": ((1, 3, 224, 224), (1, 3, 224, 224)),
        "op_params": {"op": "mul"},
    },
    {
        "expected_fpops": 150528.0,
        "expected_output_shape": (2, 3, 224, 224),
        "input_shapes": ((2, 3, 224, 224), (2, 3, 224, 224)),
        "op_params": {"op": "add"},
    },
]


@pytest.mark.parametrize("test_params", BINARY_TEST_PARAMS)
def test_calc_fpops_pool2d_nonadaptive(test_params):
    expected_fpops = test_params["expected_fpops"]
    expected_output_shape = test_params["expected_output_shape"]
    op_params = test_params["op_params"]
    input_shapes = test_params["input_shapes"]

    m = WrapBinary(**op_params)

    sample_inputs = pth.make_rand_tensors(input_shapes, torch.device("cpu"))
    fpops = pth.stats.calc_fpops(m, input_shapes=input_shapes, unit="flops")
    output = m(sample_inputs[0], sample_inputs[1])
    output_shape = tuple(output.shape)
    assert output_shape == expected_output_shape
    assert fpops == expected_fpops
