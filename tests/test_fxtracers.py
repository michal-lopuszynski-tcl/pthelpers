import pytest
import torch

import pthelpers as pth

try:
    import torchvision
except ModuleNotFoundError:
    torchvision = None

try:
    import timm
except ModuleNotFoundError:
    timm = None


def check_models_equivalence(m1: torch.nn.Module, m2: torch.nn.Module):
    x = torch.rand(2, 3, 224, 224)

    m1.eval()
    y1 = m1(x).detach()

    m2.eval()
    y2 = m2(x).detach()

    d1 = m1.state_dict()

    d2 = m2.state_dict()

    k1 = sorted(d1.keys())
    k2 = sorted(d2.keys())

    assert k1 == k2

    torch.testing.assert_close(y1, y2)


TORCHVISION_TEST_PARAMS = [
    ["resnet18", pth.fxtracers.symbolic_trace_one_level],
    ["resnet18", pth.fxtracers.symbolic_trace_nested],
    ["resnet18", pth.fxtracers.symbolic_trace_tolerant],
    ["mobilenet_v3_small", pth.fxtracers.symbolic_trace_one_level],
    ["mobilenet_v3_small", pth.fxtracers.symbolic_trace_nested],
    ["mobilenet_v3_small", pth.fxtracers.symbolic_trace_tolerant],
]


@pytest.mark.skipif(torchvision is None, reason="requires torchvision")
@pytest.mark.parametrize("model_name, tracing_fn", TORCHVISION_TEST_PARAMS)
def test_torchvision(model_name, tracing_fn):
    m1 = torchvision.models.get_model(model_name, weights=None, num_classes=10)
    m2 = tracing_fn(m1)
    check_models_equivalence(m1, m2)


TIMM_TEST_PARAMS = [
    ["inception_v4", pth.fxtracers.symbolic_trace_one_level],
    ["inception_v4", pth.fxtracers.symbolic_trace_nested],
    ["inception_v4", pth.fxtracers.symbolic_trace_tolerant],
    ["tf_efficientnet_b0", pth.fxtracers.symbolic_trace_one_level],
    ["tf_efficientnet_b0", pth.fxtracers.symbolic_trace_nested],
    ["tf_efficientnet_b0", pth.fxtracers.symbolic_trace_tolerant],
]


@pytest.mark.skipif(timm is None, reason="requires timm")
@pytest.mark.parametrize("model_name, tracing_fn", TORCHVISION_TEST_PARAMS)
def test_timm(model_name, tracing_fn):
    m1 = torchvision.models.get_model(model_name, weights=None, num_classes=10)
    m2 = tracing_fn(m1)
    check_models_equivalence(m1, m2)
