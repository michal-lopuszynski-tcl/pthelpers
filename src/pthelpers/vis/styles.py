from typing import Any, Optional, Union

import torch

from .. import core, fxfilters

NodeStyle = dict[str, Union[str, dict[str, str]]]


def _get_raw_node_name_label(node: torch.fx.Node, node_idx: int) -> str:
    return f"{node.name} idx_{node_idx}"


def _is_call_module_derived(
    node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]
) -> bool:
    # This is difficult to parse because of `torch.nn.Module` vs Python code module
    # ambiguity.
    # Function checks if "call_node" module comes from torch library.
    # If not, the module is derived from some torch classes, such as torch.nn.Conv2d

    if node.op == "call_module":
        assert isinstance(node.target, str)
        mod = module_dict[node.target]
        return not type(mod).__module__.startswith("torch.")
    return False


def _get_raw_node_label(
    node: torch.fx.Node,
    node_idx: int,
    node_result: Any,
    module_dict: dict[str, torch.nn.Module],
) -> str:
    node_op_to_abbreviation = {
        "placeholder": "PLA",
        "get_attr": "ATT",
        "call_function": "FUN",
        "call_method": "MET",
        "call_module": "MOD",
        "output": "",
    }

    label = _get_raw_node_name_label(node, node_idx)
    assert node.op in node_op_to_abbreviation
    label += "\n" + node_op_to_abbreviation[node.op] + " "
    label += core.get_fxnode_target_name(node, module_dict)
    if node_result is not None:
        label += "\n"
        if isinstance(node_result, torch.Tensor):
            label += f"{tuple(node_result.shape)} {node_result.dtype}"
        else:
            label += f"{type(node_result)}\n"
    return label


def _get_raw_node_style(
    node: torch.fx.Node,
    node_idx: int,
    node_result: Any,
    module_dict: dict[str, torch.nn.Module],
) -> NodeStyle:
    return {
        "shape": "box",
        "label": _get_raw_node_label(node, node_idx, node_result, module_dict),
    }


def get_raw_style(
    *,
    element: str,
    node_meta1: Optional[dict[str, Any]] = None,
    node_meta2: Optional[dict[str, Any]] = None,
    module_dict: Optional[dict[str, torch.nn.Module]],
) -> NodeStyle:
    if element == "node":
        assert module_dict is not None
        assert node_meta1 is not None
        return _get_raw_node_style(**node_meta1, module_dict=module_dict)
    else:
        return {}


# STD Style


def _get_default_std_style() -> dict[str, str]:
    _DEFAULT_TCL_STYLE = {
        "color": "black",
        "fillcolor": "white",
        "style": "filled",
        "shape": "box",
    }
    return _DEFAULT_TCL_STYLE


def _get_std_module_type(
    node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]
) -> str:
    assert node.op == "call_module" and isinstance(node.target, str)
    module = module_dict[node.target]
    mtype = type(module).__name__

    # This is to higlight tha modules that inherit from torch modules, such as
    # `torch.nn.Conv2d`. Since we test them via `isinstance` the might show up as
    # Conv2d, but their behaviour might be different

    mtype = mtype + " DER" if _is_call_module_derived(node, module_dict) else mtype
    return mtype


def _get_std_node_color(node_type: str) -> str:
    # NODE_API colors

    # DEFAULT_COLOR = 'white'
    # INPUT_SHAPE = 'invtriangle'
    # OUTPUT_SHAPE = 'oval'
    # INPUT_COLOR = 'green'
    # OUTPUT_COLOR = 'orange'
    # CELL_SHAPE = 'box'
    # CONV_COLOR = 'red'
    # DEP_CONV_COLOR = 'deepskyblue1'
    # ADD_COLOR = 'yellow'
    # MUL_COLOR = 'darkorchid1'
    # CONCAT_COLOR = 'beige'
    # POOLING_COLOR = 'darkgoldenrod'
    # CELL_FILL_COLOR = 'lightblue'
    # BN_COLOR = 'cornsilk'
    # ACT_COLOR = 'aquamarine3'
    # DEPTH_1_FILL_COLOR = 'gray90'
    # DEPTH_2_FILL_COLOR = 'gray80'
    # NESTED_CELL_FILL_COLOR = 'deepskyblue4'
    # NODE_SHAPE = 'box'
    # NODE_SHAPE_COLOR = 'black'
    # ATTENTION_COLOR = 'darkolivegreen4'
    # LAYER_NORM_COLOR = 'darkorange'
    # DENSE_COLOR = 'darkseagreen3'
    # MATMUL_COLOR = 'chocolate'

    _NODE_TYPE_TO_COLOR = {
        "input": "green",
        "output": "orange",
        "conv1d_plain": "tomato",
        "conv1d_dwise": "lightblue",
        "conv1d_grouped": "plum",
        "conv2d_plain": "red",
        "conv2d_dwise": "deepskyblue",
        "conv2d_grouped": "violet",
        "linear": "darkseagreen3",
        "batchnorm1d": "floralwhite",
        "batchnorm2d": "cornsilk",
        "pool": "darkgoldenrod",
        "act": "aquamarine3",
        "add": "yellow",
        "mul": "darkorchid1",
        "dropout": "pink",
        "graph_module": "deepskyblue4",
        "layernorm": "darkorange",
        "embedding": "greenyellow",
    }
    return _NODE_TYPE_TO_COLOR[node_type]


def _get_std_shape_label(node_result: Optional[tuple[int, ...]]) -> str:
    if isinstance(node_result, torch.Tensor):
        return f"\n----------\n{tuple(node_result.shape)}"
    else:
        return ""


def _get_std_default_node_style(
    node: torch.fx.Node,
    node_idx: int,
    node_result: Any,
    module_dict: dict[str, torch.nn.Module],
) -> NodeStyle:
    style = {"label": _get_raw_node_label(node, node_idx, node_result, module_dict)}
    return {**_get_default_std_style(), **style}


def _get_std_input_node_style(
    node: torch.fx.Node,
    node_idx: int,
    node_result: Any,
    module_dict: dict[str, torch.nn.Module],
) -> NodeStyle:
    label = _get_raw_node_name_label(node, node_idx)
    fillcolor = _get_std_node_color("input")
    style = {"label": label, "shape": "invtriangle", "fillcolor": fillcolor}
    return {**_get_default_std_style(), **style}


def _get_std_output_node_style(
    node: torch.fx.Node,
    node_idx: int,
    node_result: Any,
    module_dict: dict[str, torch.nn.Module],
) -> NodeStyle:
    fillcolor = _get_std_node_color("output")
    style = {"label": "output", "shape": "oval", "fillcolor": fillcolor}
    return {**_get_default_std_style(), **style}


def _get_std_conv_plain_node_style(
    node: torch.fx.Node,
    node_idx: int,
    node_result: Any,
    module_dict: dict[str, torch.nn.Module],
    n: int,
) -> NodeStyle:
    assert node.op == "call_module" and isinstance(node.target, str)
    module = module_dict[node.target]
    label = _get_raw_node_name_label(node, node_idx)
    label += f"\n{_get_std_module_type(node, module_dict)}\n----------\n"
    label += f"kernel_size: {module.kernel_size}\n"
    label += f"in_channels: {module.in_channels}\n"
    label += f"out_channels: {module.out_channels}\n"
    if module.stride != (1, 1) and module.stride != (1,):
        label += f"stride: {module.stride}\n"
    if module.dilation != (1, 1) and module.dilation != (1,):
        label += f"dilation: {module.dilation}\n"
    label += "+ bias" if module.bias is not None else "no bias"
    label += _get_std_shape_label(node_result)
    fillcolor = _get_std_node_color(f"conv{n}d_plain")
    style = {"label": label, "fillcolor": fillcolor}
    return {**_get_default_std_style(), **style}


def _get_std_conv1d_plain_node_style(
    node: torch.fx.Node,
    node_idx: int,
    node_result: Any,
    module_dict: dict[str, torch.nn.Module],
) -> NodeStyle:
    return _get_std_conv_plain_node_style(node, node_idx, node_result, module_dict, 1)


def _get_std_conv2d_plain_node_style(
    node: torch.fx.Node,
    node_idx: int,
    node_result: Any,
    module_dict: dict[str, torch.nn.Module],
) -> NodeStyle:
    return _get_std_conv_plain_node_style(node, node_idx, node_result, module_dict, 2)


def _get_std_conv_dwise_node_style(
    node: torch.fx.Node,
    node_idx: int,
    node_result: Any,
    module_dict: dict[str, torch.nn.Module],
    n: int,
) -> NodeStyle:
    assert node.op == "call_module" and isinstance(node.target, str)
    module = module_dict[node.target]
    label = _get_raw_node_name_label(node, node_idx)
    label += f"\ndwise {_get_std_module_type(node, module_dict)}\n----------\n"
    label += f"kernel_size: {module.kernel_size}\n"
    label += f"channels: {module.in_channels}\n"
    if module.stride != (1, 1) and module.stride != (1,):
        label += f"stride: {module.stride}\n"
    if module.dilation != (1, 1) and module.dilation != (1,):
        label += f"dilation: {module.dilation}\n"
    assert isinstance(module.out_channels, int) and isinstance(module.in_channels, int)
    exp_rate = int(module.out_channels / module.in_channels)
    label += f"exp_rate: {exp_rate}\n"
    label += "+ bias" if module.bias is not None else "no bias"
    label += _get_std_shape_label(node_result)
    fillcolor = _get_std_node_color(f"conv{n}d_dwise")
    style = {"label": label, "fillcolor": fillcolor}
    return {**_get_default_std_style(), **style}


def _get_std_conv1d_dwise_node_style(
    node: torch.fx.Node,
    node_idx: int,
    node_result: Any,
    module_dict: dict[str, torch.nn.Module],
) -> NodeStyle:
    return _get_std_conv_dwise_node_style(node, node_idx, node_result, module_dict, 1)


def _get_std_conv2d_dwise_node_style(
    node: torch.fx.Node,
    node_idx: int,
    node_result: Any,
    module_dict: dict[str, torch.nn.Module],
) -> NodeStyle:
    return _get_std_conv_dwise_node_style(node, node_idx, node_result, module_dict, 2)


def _get_std_conv_grouped_node_style(
    node: torch.fx.Node,
    node_idx: int,
    node_result: Any,
    module_dict: dict[str, torch.nn.Module],
    n: int,
) -> NodeStyle:
    assert node.op == "call_module" and isinstance(node.target, str)
    module = module_dict[node.target]
    label = _get_raw_node_name_label(node, node_idx)
    label += f"\ngrouped  {_get_std_module_type(node, module_dict)}\n----------\n"
    label += f"kernel_size: {module.kernel_size}\n"
    label += f"in_channels: {module.in_channels}\n"
    label += f"out_channels: {module.out_channels}\n"
    if module.stride != (1, 1) and module.stride != (1,):
        label += f"stride: {module.stride}\n"
    if module.dilation != (1, 1) and module.dilation != (1,):
        label += f"dilation: {module.dilation}\n"
    label += f"groups: {module.groups}\n"
    label += "+ bias" if module.bias is not None else "no bias"
    label += _get_std_shape_label(node_result)
    fillcolor = _get_std_node_color(f"conv{n}d_grouped")
    style = {"label": label, "fillcolor": fillcolor}
    return {**_get_default_std_style(), **style}


def _get_std_conv1d_grouped_node_style(
    node: torch.fx.Node,
    node_idx: int,
    node_result: Any,
    module_dict: dict[str, torch.nn.Module],
) -> NodeStyle:
    return _get_std_conv_grouped_node_style(node, node_idx, node_result, module_dict, 1)


def _get_std_conv2d_grouped_node_style(
    node: torch.fx.Node,
    node_idx: int,
    node_result: Any,
    module_dict: dict[str, torch.nn.Module],
) -> NodeStyle:
    return _get_std_conv_grouped_node_style(node, node_idx, node_result, module_dict, 2)


def _get_std_linear_node_style(
    node: torch.fx.Node,
    node_idx: int,
    node_result: Any,
    module_dict: dict[str, torch.nn.Module],
) -> NodeStyle:
    assert node.op == "call_module" and isinstance(node.target, str)
    module = module_dict[node.target]
    label = _get_raw_node_name_label(node, node_idx)
    label += f"\n{_get_std_module_type(node, module_dict)}\n----------\n"
    label += f"in_features: {module.in_features}\n"
    label += f"out_features: {module.out_features}\n"
    label += "+ bias" if module.bias is not None else "no bias"
    label += _get_std_shape_label(node_result)
    fillcolor = _get_std_node_color("linear")
    style = {"label": label, "fillcolor": fillcolor}
    return {**_get_default_std_style(), **style}


def _get_std_batchnorm_node_style(
    node: torch.fx.Node,
    node_idx: int,
    node_result: Any,
    module_dict: dict[str, torch.nn.Module],
    n: int,
) -> NodeStyle:
    assert node.op == "call_module" and isinstance(node.target, str)
    label = _get_raw_node_name_label(node, node_idx)
    label += f"\n{_get_std_module_type(node, module_dict)}"
    label += _get_std_shape_label(node_result)
    fillcolor = _get_std_node_color(f"batchnorm{n}d")
    style = {"label": label, "fillcolor": fillcolor}
    return {**_get_default_std_style(), **style}


def _get_std_batchnorm1d_node_style(
    node: torch.fx.Node,
    node_idx: int,
    node_result: Any,
    module_dict: dict[str, torch.nn.Module],
) -> NodeStyle:
    return _get_std_batchnorm_node_style(node, node_idx, node_result, module_dict, 2)


def _get_std_batchnorm2d_node_style(
    node: torch.fx.Node,
    node_idx: int,
    node_result: Any,
    module_dict: dict[str, torch.nn.Module],
) -> NodeStyle:
    return _get_std_batchnorm_node_style(node, node_idx, node_result, module_dict, 2)


def _get_std_layernorm_node_style(
    node: torch.fx.Node,
    node_idx: int,
    node_result: Any,
    module_dict: dict[str, torch.nn.Module],
) -> NodeStyle:
    assert node.op == "call_module" and isinstance(node.target, str)
    label = _get_raw_node_name_label(node, node_idx)
    label += f"\n{_get_std_module_type(node, module_dict)}"
    label += _get_std_shape_label(node_result)
    fillcolor = _get_std_node_color("layernorm")
    style = {"label": label, "fillcolor": fillcolor}
    return {**_get_default_std_style(), **style}


def _get_std_pool_node_style(
    node: torch.fx.Node,
    node_idx: int,
    node_result: Any,
    module_dict: dict[str, torch.nn.Module],
) -> NodeStyle:
    assert node.op == "call_module" and isinstance(node.target, str)
    label = _get_raw_node_name_label(node, node_idx)
    label += f"\n{_get_std_module_type(node, module_dict)}"
    label += _get_std_shape_label(node_result)
    fillcolor = _get_std_node_color("pool")
    style = {"label": label, "fillcolor": fillcolor}
    return {**_get_default_std_style(), **style}


def _get_std_act_node_style(
    node: torch.fx.Node,
    node_idx: int,
    node_result: Any,
    module_dict: dict[str, torch.nn.Module],
) -> NodeStyle:
    if node.op == "call_module":
        assert isinstance(node.target, str)
        module = module_dict[node.target]
        activation_name = type(module).__name__
    elif node.op == "call_function":
        assert callable(node.target)
        activation_name = node.target.__name__
    else:
        assert False, "Activation can be either module or function"
    label = _get_raw_node_name_label(node, node_idx)
    label += f"\nAct {activation_name}"
    label += _get_std_shape_label(node_result)
    fillcolor = _get_std_node_color("act")
    style = {"label": label, "fillcolor": fillcolor}
    return {**_get_default_std_style(), **style}


def _get_std_add_node_style(
    node: torch.fx.Node,
    node_idx: int,
    node_result: Any,
    module_dict: dict[str, torch.nn.Module],
) -> NodeStyle:
    label = _get_raw_node_name_label(node, node_idx)
    label += "\n+"
    fillcolor = _get_std_node_color("add")
    style = {"label": label, "fillcolor": fillcolor}
    return {**_get_default_std_style(), **style}


def _get_std_mul_node_style(
    node: torch.fx.Node,
    node_idx: int,
    node_result: Any,
    module_dict: dict[str, torch.nn.Module],
) -> NodeStyle:
    label = _get_raw_node_name_label(node, node_idx)
    label += "\n*"
    fillcolor = _get_std_node_color("mul")
    style = {"label": label, "fillcolor": fillcolor}
    return {**_get_default_std_style(), **style}


def _get_std_dropout_node_style(
    node: torch.fx.Node,
    node_idx: int,
    node_result: Any,
    module_dict: dict[str, torch.nn.Module],
) -> NodeStyle:
    label = _get_raw_node_name_label(node, node_idx)
    label += "\nDropout"
    label += _get_std_shape_label(node_result)
    fillcolor = _get_std_node_color("dropout")
    style = {"label": label, "fillcolor": fillcolor}
    return {**_get_default_std_style(), **style}


def _get_std_embedding_style(
    node: torch.fx.Node,
    node_idx: int,
    node_result: Any,
    module_dict: dict[str, torch.nn.Module],
) -> NodeStyle:
    assert node.op == "call_module" and isinstance(node.target, str)
    module = module_dict[node.target]
    assert isinstance(module, torch.nn.Embedding)
    label = _get_raw_node_name_label(node, node_idx)
    label += f"\n{_get_std_module_type(node, module_dict)}\n----------\n"
    label += f"num_embeddings: {module.num_embeddings}\n"
    label += f"embedding_dim: {module.embedding_dim}"
    label += _get_std_shape_label(node_result)
    fillcolor = _get_std_node_color("embedding")
    style = {"label": label, "fillcolor": fillcolor}
    return {**_get_default_std_style(), **style}


def _get_std_graph_module_style(
    node: torch.fx.Node,
    node_idx: int,
    node_result: Any,
    module_dict: dict[str, torch.nn.Module],
) -> NodeStyle:
    label = _get_raw_node_name_label(node, node_idx)
    label += f"\n{_get_std_module_type(node, module_dict)}\n----------\n"
    fillcolor = _get_std_node_color("graph_module")
    style = {"label": label, "fillcolor": fillcolor}
    return {**_get_default_std_style(), **style}


def _add_metadata_fillcolor(style: NodeStyle, node: torch.fx.Node) -> None:
    meta_pth_vis = node.meta.get("pth_vis")
    if isinstance(meta_pth_vis, dict):
        fillcolor = meta_pth_vis.get("fillcolor")
        if fillcolor is not None:
            if not isinstance(fillcolor, str):
                msg = f"{node.name=} node.meta['pth_vis']['fillcolor'] should be a str"
                raise ValueError(msg)
            style["fillcolor"] = fillcolor


def _add_metadata_label(style: NodeStyle, node: torch.fx.Node) -> None:
    meta_pth_vis = node.meta.get("pth_vis")
    if isinstance(meta_pth_vis, dict):
        label = meta_pth_vis.get("label")
        if label is not None:
            if not isinstance(label, str):
                msg = f"{node.name=} node.meta['pth_vis']['label'] should be a str"
                raise ValueError(msg)
            old_label = style["label"]
            assert isinstance(old_label, str)
            style["label"] = old_label + "\n" + label


def _get_std_node_style(
    node: torch.fx.Node,
    node_idx: int,
    node_result: Any,
    module_dict: dict[str, torch.nn.Module],
) -> NodeStyle:
    tcl_style = [
        (fxfilters.is_conv2d_plain, _get_std_conv2d_plain_node_style),
        (fxfilters.is_conv2d_dwise, _get_std_conv2d_dwise_node_style),
        (fxfilters.is_conv2d_grouped, _get_std_conv2d_grouped_node_style),
        (fxfilters.is_conv1d_plain, _get_std_conv1d_plain_node_style),
        (fxfilters.is_conv1d_dwise, _get_std_conv1d_dwise_node_style),
        (fxfilters.is_conv1d_grouped, _get_std_conv1d_grouped_node_style),
        (fxfilters.is_batchnorm1d, _get_std_batchnorm1d_node_style),
        (fxfilters.is_batchnorm2d, _get_std_batchnorm2d_node_style),
        (fxfilters.is_layernorm, _get_std_layernorm_node_style),
        (fxfilters.is_linear, _get_std_linear_node_style),
        (fxfilters.is_activation, _get_std_act_node_style),
        (fxfilters.is_pool, _get_std_pool_node_style),
        (fxfilters.is_add, _get_std_add_node_style),
        (fxfilters.is_mul, _get_std_mul_node_style),
        (fxfilters.is_input, _get_std_input_node_style),
        (fxfilters.is_output, _get_std_output_node_style),
        (fxfilters.is_dropout, _get_std_dropout_node_style),
        (fxfilters.is_embedding, _get_std_embedding_style),
        (fxfilters.is_graph_module, _get_std_graph_module_style),
        (fxfilters.is_any, _get_std_default_node_style),
    ]

    for matches_fn, get_style_fn in tcl_style:
        if matches_fn(node, module_dict):
            style = get_style_fn(node, node_idx, node_result, module_dict)
            _add_metadata_fillcolor(style, node)
            _add_metadata_label(style, node)
            return style

    assert False
    # Replace default fillcolor by the one in metadata if present


def get_std_style(
    *,
    element: str,
    node_meta1: Optional[dict[str, Any]] = None,
    node_meta2: Optional[dict[str, Any]] = None,
    module_dict: Optional[dict[str, torch.nn.Module]] = None,
) -> NodeStyle:
    if element == "node":
        assert node_meta1 is not None
        assert module_dict is not None
        return _get_std_node_style(**node_meta1, module_dict=module_dict)
    else:
        return {}


# MIN Style

DEFAULT_STD_MIN_STYLE = {
    "label": "",
    "color": "black",
    "fillcolor": "white",
    "style": "filled",
    "shape": "box",
    "fixedsize": "True",
    "width": "0.2",
    "height": "0.2",
}


def _get_std_min_node_style(
    node: torch.fx.Node,
    node_idx: int,
    node_result: Any,
    module_dict: dict[str, torch.nn.Module],
) -> NodeStyle:
    tcl_style = _get_std_node_style(node, node_idx, node_result, module_dict)

    return {
        **DEFAULT_STD_MIN_STYLE,
        **{
            "label": "",
            "tooltip": tcl_style["label"],
            "fillcolor": tcl_style["fillcolor"],
        },
    }


def get_std_min_style(
    *,
    element: str,
    node_meta1: Optional[dict[str, Any]] = None,
    node_meta2: Optional[dict[str, Any]] = None,
    module_dict: Optional[dict[str, torch.nn.Module]] = None,
) -> NodeStyle:
    if element == "node":
        assert node_meta1 is not None
        assert module_dict is not None
        return _get_std_min_node_style(**node_meta1, module_dict=module_dict)
    elif element == "edge":
        return {"arrowsize": "0.5"}
    elif element == "graph":
        return {"graph_attr": {"ranksep": "0.1", "nodesep": "0.1"}}
    else:
        return {}
