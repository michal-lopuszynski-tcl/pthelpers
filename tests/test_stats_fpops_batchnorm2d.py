import pytest
import torch

import pthelpers as pth


class WrapBatchNorm2d(torch.nn.Module):
    def __init__(self, *, num_features):
        super().__init__()
        self.op = torch.nn.BatchNorm2d(num_features=num_features)

    def forward(self, x):
        return self.op(x)


BATCHNORM2D_TEST_PARAMS = [
    {
        "expected_fpops": 602112.0,
        "expected_output_shape": (1, 3, 224, 224),
        "input_shape": (1, 3, 224, 224),
        "op_params": {"num_features": 3},
    },
    {
        "expected_fpops": 802816.0,
        "expected_output_shape": (1, 16, 112, 112),
        "input_shape": (1, 16, 112, 112),
        "op_params": {"num_features": 16},
    },
    {
        "expected_fpops": 12845056.0,
        "expected_output_shape": (1, 64, 224, 224),
        "input_shape": (1, 64, 224, 224),
        "op_params": {"num_features": 64},
    },
    {
        "expected_fpops": 602112.0,
        "expected_output_shape": (3, 3, 224, 224),
        "input_shape": (3, 3, 224, 224),
        "op_params": {"num_features": 3},
    },
    {
        "expected_fpops": 802816.0,
        "expected_output_shape": (4, 16, 112, 112),
        "input_shape": (4, 16, 112, 112),
        "op_params": {"num_features": 16},
    },
    {
        "expected_fpops": 12845056.0,
        "expected_output_shape": (2, 64, 224, 224),
        "input_shape": (2, 64, 224, 224),
        "op_params": {"num_features": 64},
    },
]


@pytest.mark.parametrize("test_params", BATCHNORM2D_TEST_PARAMS)
def test_calc_fpops_batchnorm2d(test_params):
    expected_fpops = test_params["expected_fpops"]
    expected_output_shape = test_params["expected_output_shape"]
    op_params = test_params["op_params"]
    input_shape = test_params["input_shape"]
    m = WrapBatchNorm2d(**op_params)
    sample_inputs = pth.make_rand_tensors((input_shape,), torch.device("cpu"))
    fpops = pth.stats.calc_fpops(m, input_shapes=(input_shape,), unit="flops")
    output = m(sample_inputs[0])
    output_shape = tuple(output.shape)
    assert output_shape == expected_output_shape
    assert fpops == expected_fpops
