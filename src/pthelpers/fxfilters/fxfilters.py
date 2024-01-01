import operator

import torch

__all__ = [
    "get_nn_op_type",
    "is_activation",
    "is_add",
    "is_any",
    "is_batchnorm1d",
    "is_batchnorm2d",
    "is_binary_op",
    "is_cat",
    "is_cat_channels",
    "is_cat_not_channels",
    "is_conv1d",
    "is_conv1d_dwise",
    "is_conv1d_grouped",
    "is_conv1d_plain",
    "is_conv2d",
    "is_conv2d_dwise",
    "is_conv2d_grouped",
    "is_conv2d_plain",
    "is_conv2d_plain_1x1",
    "is_conv2d_plain_non1x1",
    "is_dropout",
    "is_embedding",
    "is_flatten",
    "is_get_attr",
    "is_graph_module",
    "is_input",
    "is_layernorm",
    "is_linear",
    "is_mul",
    "is_output",
    "is_pool",
    "is_pool2d_adaptive_avg",
    "is_pool2d_adaptive_max",
    "is_pool2d_avg",
    "is_pool2d_max",
    "is_with_args",
    "is_with_params",
    "is_without_args",
    "is_without_params",
]


def is_input(node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]) -> bool:
    return node.op == "placeholder"


def is_output(node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]) -> bool:
    return node.op == "output"


def is_get_attr(node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]) -> bool:
    return node.op == "get_attr"


def is_activation(node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]) -> bool:
    activation_funs = [
        torch.nn.functional.relu,
        torch.nn.functional.hardtanh,
        torch.nn.functional.relu6,
        torch.nn.functional.selu,
        torch.nn.functional.celu,
        torch.nn.functional.leaky_relu,
        torch.nn.functional.prelu,
        torch.nn.functional.rrelu,
        torch.nn.functional.glu,
        torch.nn.functional.gelu,
        torch.nn.functional.softsign,
        torch.nn.functional.softplus,
        torch.nn.functional.softmin,
        torch.nn.functional.softmax,
        torch.nn.functional.log_softmax,
        torch.nn.functional.tanh,
        torch.nn.functional.sigmoid,
        torch.nn.functional.hardsigmoid,
        torch.nn.functional.silu,
        torch.nn.functional.mish,
        torch.sigmoid,
    ]

    if node.op == "call_module":
        module = module_dict[str(node.target)]
        return module.__module__ == "torch.nn.modules.activation"
    elif node.op == "call_function":
        return node.target in activation_funs
    return False


def is_batchnorm1d(
    node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]
) -> bool:
    if node.op == "call_module":
        module = module_dict[str(node.target)]
        return isinstance(module, torch.nn.BatchNorm1d)
    return False


def is_batchnorm2d(
    node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]
) -> bool:
    if node.op == "call_module":
        module = module_dict[str(node.target)]
        return isinstance(module, torch.nn.BatchNorm2d)
    return False


def is_layernorm(node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]) -> bool:
    if node.op == "call_module":
        module = module_dict[str(node.target)]
        return isinstance(module, torch.nn.LayerNorm)
    return False


def is_conv1d(node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]) -> bool:
    if node.op == "call_module":
        module = module_dict[str(node.target)]
        return isinstance(module, torch.nn.Conv1d)
    else:
        return False


def is_conv1d_plain(
    node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]
) -> bool:
    if node.op == "call_module":
        module = module_dict[str(node.target)]
        return isinstance(module, torch.nn.Conv1d) and module.groups == 1
    else:
        return False


def is_conv1d_dwise(
    node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]
) -> bool:
    if node.op == "call_module":
        module = module_dict[str(node.target)]
        return (
            isinstance(module, torch.nn.Conv1d)
            and module.groups == module.in_channels
            and module.out_channels % module.in_channels == 0
        )
    else:
        return False


def is_conv1d_grouped(
    node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]
) -> bool:
    if node.op == "call_module":
        module = module_dict[str(node.target)]
        if isinstance(module, torch.nn.Conv1d):
            return module.groups != 1 and not (
                module.groups == module.in_channels
                and module.out_channels % module.in_channels == 0
            )
    return False


def is_conv2d(node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]) -> bool:
    if node.op == "call_module":
        module = module_dict[str(node.target)]
        return isinstance(module, torch.nn.Conv2d)
    else:
        return False


def is_conv2d_plain(
    node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]
) -> bool:
    if node.op == "call_module":
        module = module_dict[str(node.target)]
        return isinstance(module, torch.nn.Conv2d) and module.groups == 1
    else:
        return False


def is_conv2d_plain_1x1(
    node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]
) -> bool:
    if node.op == "call_module":
        module = module_dict[str(node.target)]
        return (
            isinstance(module, torch.nn.Conv2d)
            and module.groups == 1
            and module.kernel_size == (1, 1)
        )
    else:
        return False


def is_conv2d_plain_non1x1(
    node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]
) -> bool:
    if node.op == "call_module":
        module = module_dict[str(node.target)]
        return (
            isinstance(module, torch.nn.Conv2d)
            and module.groups == 1
            and module.kernel_size != (1, 1)
        )
    else:
        return False


def is_conv2d_dwise(
    node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]
) -> bool:
    if node.op == "call_module":
        module = module_dict[str(node.target)]
        return (
            isinstance(module, torch.nn.Conv2d)
            and module.groups == module.in_channels
            and module.out_channels % module.in_channels == 0
        )
    else:
        return False


def is_conv2d_grouped(
    node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]
) -> bool:
    if node.op == "call_module":
        module = module_dict[str(node.target)]
        if isinstance(module, torch.nn.Conv2d):
            return module.groups != 1 and not (
                module.groups == module.in_channels
                and module.out_channels % module.in_channels == 0
            )
    return False


def is_linear(node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]) -> bool:
    if node.op == "call_module":
        module = module_dict[str(node.target)]
        return isinstance(module, torch.nn.Linear)
    else:
        return False


def is_pool(node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]) -> bool:
    # TODO Compose this out of simpler is_
    if node.op == "call_module":
        module = module_dict[str(node.target)]
        return module.__module__ == "torch.nn.modules.pooling"
    return False


def is_pool2d_avg(node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]) -> bool:
    if node.op == "call_module":
        module = module_dict[str(node.target)]
        return isinstance(module, torch.nn.AvgPool2d)
    elif node.op == "call_function":
        return node.target is torch.nn.functional.avg_pool2d
    return False


def is_pool2d_max(node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]) -> bool:
    if node.op == "call_module":
        module = module_dict[str(node.target)]
        return isinstance(module, torch.nn.MaxPool2d)
    elif node.op == "call_function":
        return node.target is torch.nn.functional.max_pool2d
    return False


def is_pool2d_adaptive_avg(
    node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]
) -> bool:
    if node.op == "call_module":
        module = module_dict[str(node.target)]
        return isinstance(module, torch.nn.AdaptiveAvgPool2d)
    elif node.op == "call_function":
        return node.target is torch.nn.functional.adaptive_avg_pool2d
    return False


def is_pool2d_adaptive_max(
    node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]
) -> bool:
    if node.op == "call_module":
        module = module_dict[str(node.target)]
        return isinstance(module, torch.nn.AdaptiveMaxPool2d)
    elif node.op == "call_function":
        return node.target is torch.nn.functional.adaptive_max_pool2d
    return False


def is_add(node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]) -> bool:
    return node.op == "call_function" and node.target is operator.add


def is_mul(node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]) -> bool:
    return node.op == "call_function" and node.target is operator.mul


def is_binary_op(node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]) -> bool:
    return is_mul(node, module_dict) or is_add(node, module_dict)


def is_cat(node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]) -> bool:
    return node.op == "call_function" and node.target is torch.cat


def is_cat_channels(
    node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]
) -> bool:
    # TODO: Unify kwargs and args here, so it works regardless of call
    # Type checker complains if this is implemented as `return condintion`
    if is_cat(node, module_dict) and node.kwargs["dim"] == 1:
        return True
    else:
        return False


def is_cat_not_channels(
    node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]
) -> bool:
    # TODO: Unify kwargs and args here, so it works regardless of call
    # Type checker complains if this is implemented as `return condintion`
    if is_cat(node, module_dict) and node.kwargs["dim"] != 1:
        return True
    else:
        return False


def is_flatten(node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]) -> bool:
    # TODO Add module version
    if node.op == "call_function" and node.target is torch.flatten:
        return True
    elif node.op == "call_method" and node.target == "flatten":
        # TODO Should we add type checking?, flatten might not be a tensor method
        return True
    else:
        return False


def is_dropout(node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]) -> bool:
    # TODO Add functional version
    return node.op == "call_module" and isinstance(
        module_dict[str(node.target)], torch.nn.Dropout
    )


def is_embedding(node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]) -> bool:
    return node.op == "call_module" and isinstance(
        module_dict[str(node.target)], torch.nn.Embedding
    )


def is_with_args(node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]) -> bool:
    return len(node.all_input_nodes) != 0


def is_without_args(
    node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]
) -> bool:
    return not is_with_args(node, module_dict)


def is_with_params(
    node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]
) -> bool:
    if node.op == "call_module":
        assert isinstance(node.target, str)
        module_params = module_dict[node.target].parameters()
        for p in module_params:
            return True
        return False
    else:
        return False


def is_without_params(
    node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]
) -> bool:
    return not is_with_params(node, module_dict)


def is_any(node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]) -> bool:
    return True


def is_graph_module(
    node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]
) -> bool:
    if node.op == "call_module":
        assert isinstance(node.target, str)
        m = module_dict[node.target]
        return isinstance(m, torch.fx.GraphModule)
    return False


def get_nn_op_type(node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]) -> str:
    """
    Helper function useful for fpops or param splitting
    """
    if is_conv2d_plain_non1x1(node, module_dict):
        return "conv2d_plain_non1x1"
    elif is_conv2d_plain_1x1(node, module_dict):
        return "conv2d_plain_1x1"
    elif is_conv2d_dwise(node, module_dict):
        return "conv2d_dwise"
    elif is_conv2d_grouped(node, module_dict):
        return "conv2d_grouped"
    elif is_linear(node, module_dict):
        return "linear"
    elif is_batchnorm2d(node, module_dict):
        return "batchnorm2d"
    elif is_activation(node, module_dict):
        return "activation"
    elif is_add(node, module_dict):
        return "add"
    elif is_mul(node, module_dict):
        return "mul"
    elif is_flatten(node, module_dict):
        return "flatten"
    elif is_input(node, module_dict):
        return "input"
    elif is_output(node, module_dict):
        return "output"
    else:
        return "other"
