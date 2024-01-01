import collections.abc

import torch
import torch.fx.passes

from .. import core, fxfilters

__all__ = [
    "calc_fpops",
    "calc_fpops_add_mul",
    "calc_fpops_batchnorm2d",
    "calc_fpops_conv2d_plain",
    "calc_fpops_conv2d_dwise",
    "calc_fpops_linear",
    "calc_fpops_pool2d",
    "calc_fpops_pool2d_adaptive",
    "calc_fpops_split",
    "get_fpops_divisor",
    "make_calc_fpops_pass",
]


def _get_shape_t(node: torch.fx.Node) -> torch.Tensor:
    return torch.tensor(node.meta["tensor_meta"].shape)


def calc_fpops_linear(
    node: torch.fx.Node,
    module_dict: dict[str, torch.nn.Module],
    input_shapes: list[torch.Tensor],
    output_shape: torch.Tensor,
) -> torch.Tensor:
    linear = module_dict[str(node.target)]
    input_shape = input_shapes[0]
    in_features = input_shape[-1]
    out_features = output_shape[-1]
    additional_axes_multiplier = torch.prod(input_shape[1:-1])
    per_row = 2.0 * in_features
    if linear.bias is not None:
        per_row += 1
    return out_features * per_row * additional_axes_multiplier


def calc_fpops_conv2d_plain(
    node: torch.fx.Node,
    module_dict: dict[str, torch.nn.Module],
    input_shapes: list[torch.Tensor],
    output_shape: torch.Tensor,
) -> torch.Tensor:
    conv = module_dict[str(node.target)]
    assert isinstance(conv, torch.nn.Conv2d)
    kernel_h, kernel_w = conv.kernel_size
    input_shape = input_shapes[0]
    per_kernel = 2 * kernel_h * kernel_w * input_shape[1] - 1
    output_volume = torch.prod(output_shape[1:])
    if conv.bias is not None:
        per_kernel = per_kernel + 1
    fpops = output_volume * per_kernel
    return fpops


def calc_fpops_conv2d_dwise(
    node: torch.fx.Node,
    module_dict: dict[str, torch.nn.Module],
    input_shapes: list[torch.Tensor],
    output_shape: torch.Tensor,
) -> torch.Tensor:
    # per_depthwise_kernel = 2 * kernel_h * kernel_w - 1
    # if has_bias:
    #     per_depthwise_kernel = per_depthwise_kernel + 1
    # output_volume = tf.reduce_prod(average_output_shape)
    # return tf.cast(output_volume, 'float32') * tf.cast(per_depthwise_kernel,'float32')

    conv = module_dict[str(node.target)]
    assert isinstance(conv, torch.nn.Conv2d)
    kernel_h, kernel_w = conv.kernel_size
    per_depthwise_kernel = 2 * kernel_h * kernel_w - 1
    output_volume = torch.prod(output_shape[1:])
    if conv.bias is not None:
        per_depthwise_kernel += 1
    fpops = output_volume * per_depthwise_kernel
    return fpops


def calc_fpops_pool2d(
    node: torch.fx.Node,
    module_dict: dict[str, torch.nn.Module],
    input_shapes: list[torch.Tensor],
    output_shape: torch.Tensor,
) -> torch.Tensor:
    # TODO Fix functional case
    pool = module_dict[str(node.target)]
    assert isinstance(pool, (torch.nn.MaxPool2d, torch.nn.AvgPool2d))
    if isinstance(pool.kernel_size, int):
        kernel_h, kernel_w = pool.kernel_size, pool.kernel_size
    elif isinstance(pool.kernel_size, tuple):
        kernel_h, kernel_w = pool.kernel_size
    output_volume = torch.prod(output_shape[1:])
    # TODO Should we add + 1 for division in avg case
    fpops = output_volume * kernel_h * kernel_w
    return fpops


def calc_fpops_pool2d_adaptive(
    node: torch.fx.Node,
    module_dict: dict[str, torch.nn.Module],
    input_shapes: list[torch.Tensor],
    output_shape: torch.Tensor,
) -> torch.Tensor:
    kernel = torch.div(input_shapes[0][2:], output_shape[2:])
    per_kernel = torch.prod(kernel)
    output_volume = torch.prod(output_shape[1:])
    # TODO Should we add + 1 for division in avg case
    return per_kernel * output_volume


def calc_fpops_batchnorm2d(
    node: torch.fx.Node,
    module_dict: dict[str, torch.nn.Module],
    input_shapes: list[torch.Tensor],
    output_shape: torch.Tensor,
) -> torch.Tensor:
    return 4.0 * torch.prod(output_shape[1:])


def calc_fpops_add_mul(
    node: torch.fx.Node,
    module_dict: dict[str, torch.nn.Module],
    input_shapes: list[torch.Tensor],
    output_shape: torch.Tensor,
) -> torch.Tensor:
    return torch.prod(output_shape[1:])


def calc_fpops_unknown(
    node: torch.fx.Node,
    module_dict: dict[str, torch.nn.Module],
    input_shapes: list[torch.Tensor],
    output_shape: torch.Tensor,
) -> torch.Tensor:
    return torch.tensor(-1, dtype=torch.int64)


def calc_node_fpops(
    node: torch.fx.Node,
    module_dict: dict[str, torch.nn.Module],
    input_shapes_t: list[torch.Tensor],
    output_shape_t: torch.Tensor,
) -> torch.Tensor:
    strategy = [
        (fxfilters.is_conv2d_plain, calc_fpops_conv2d_plain),
        (fxfilters.is_conv2d_dwise, calc_fpops_conv2d_dwise),
        (fxfilters.is_linear, calc_fpops_linear),
        (fxfilters.is_batchnorm2d, calc_fpops_batchnorm2d),
        (fxfilters.is_add, calc_fpops_add_mul),
        (fxfilters.is_mul, calc_fpops_add_mul),
        (fxfilters.is_pool2d_avg, calc_fpops_pool2d),
        (fxfilters.is_pool2d_max, calc_fpops_pool2d),
        (fxfilters.is_pool2d_adaptive_avg, calc_fpops_pool2d_adaptive),
        (fxfilters.is_pool2d_adaptive_max, calc_fpops_pool2d_adaptive),
        (fxfilters.is_any, calc_fpops_unknown),
    ]
    for matcher_fn, calc_fpops_fn in strategy:
        if matcher_fn(node, module_dict):
            return calc_fpops_fn(node, module_dict, input_shapes_t, output_shape_t)

    assert False, f"Unhandled node type {node.name}"


def make_calc_fpops_pass(
    gm: torch.fx.GraphModule,
    input_shapes: tuple[tuple[int, ...]],
    calc_node_fpops_fn: collections.abc.Callable[..., torch.Tensor] = calc_node_fpops,
) -> None:
    # TODO Idea - check if fpops pass has not occurred already, perhaps force option?
    device = core.get_device(gm)
    sample_inputs = core.make_rand_tensors(input_shapes, device=device)
    torch.fx.passes.shape_prop.ShapeProp(gm).propagate(*sample_inputs)
    module_dict = dict(gm.named_modules())

    for node in gm.graph.nodes:
        input_shapes_t = [_get_shape_t(pred) for pred in node.all_input_nodes]
        output_shape_t = _get_shape_t(node)
        fpops_t = calc_node_fpops_fn(node, module_dict, input_shapes_t, output_shape_t)
        node.meta["fpops"] = fpops_t.item()


def get_fpops_divisor(unit: str, input_shapes: tuple[tuple[int, ...], ...]) -> float:
    # TODO Add warning on kmapps & multiple inputs len(input_shapes) > 1
    unit = unit.lower()
    if unit == "flops" or unit == "multiadds":
        return 1
    elif unit == "gflops":
        return 1.0e9
    elif unit == "macs":
        # This is very approximate
        return 2.0
    elif unit == "gmacs":
        # This is very approximate
        return 2.0e9
    elif unit == "kmapps":
        num_pixels = 1
        for px in input_shapes[0][2:]:
            num_pixels *= px

        return 1000 * num_pixels
    else:
        raise ValueError(f'Unknown fpops unit "{unit}"')


def calc_fpops(
    m: torch.nn.Module,
    input_shapes: tuple[tuple[int, ...]],
    calc_node_fpops_fn: collections.abc.Callable[..., torch.Tensor] = calc_node_fpops,
    unit: str = "kmapps",
) -> float:
    # * _fixed have batch dimension=1, regardless of a given inputs

    gm = core.symbolic_trace_if_needed(m)

    make_calc_fpops_pass(gm, input_shapes, calc_node_fpops_fn)

    fpops = 0

    for node in gm.graph.nodes:
        fpops += max(0.0, node.meta["fpops"])

    return fpops / get_fpops_divisor(unit, input_shapes)


def calc_fpops_split(
    m: torch.nn.Module,
    input_shapes: tuple[tuple[int, ...]],
    *,
    calc_node_fpops_fn: collections.abc.Callable[..., torch.Tensor] = calc_node_fpops,
    split_fn: collections.abc.Callable[
        [torch.fx.Node, dict[str, torch.nn.Module]], str
    ] = core.get_fxnode_name,
    unit: str = "flops",
    split_unsupported: bool = False,
    sort_by: str = "fpops_r",
) -> dict[str, tuple[float, int]]:
    res = {"unsupported": (0.0, 0)}

    sort_by_to_key_fn = {
        "fpops": lambda key: res[key][0],
        "count": lambda key: res[key][1],
        "fpops_r": lambda key: -res[key][0],
        "count_r": lambda key: -res[key][1],
    }

    if sort_by not in sort_by_to_key_fn:
        raise ValueError(f"{sort_by=} not in {list(sort_by_to_key_fn.keys())}")

    gm = core.symbolic_trace_if_needed(m)
    make_calc_fpops_pass(gm, input_shapes, calc_node_fpops_fn)

    module_dict = dict(gm.named_modules())

    for node in gm.graph.nodes:
        fpops = node.meta["fpops"]
        if split_unsupported and fpops < 0:
            key = "unsupported"
            fpops = 0
        else:
            key = split_fn(node, module_dict)
            fpops = max(0, fpops)
        if key in res:
            old_fpops, old_count = res[key]
            res[key] = (old_fpops + fpops, old_count + 1)
        else:
            res[key] = (float(fpops), 1)

    divisor = get_fpops_divisor(unit, input_shapes)

    res_sorted = {}
    for key in sorted(res.keys(), key=sort_by_to_key_fn[sort_by]):
        old_fpops, old_count = res[key]
        res_sorted[key] = (old_fpops / divisor, old_count)

    return res_sorted
