import collections

import torch

from .. import core, fxfilters

__all__ = [
    "calc_params",
    "calc_params_split",
]


def _get_params_divisor(unit: str) -> float:
    unit = unit.lower()
    if unit == "params":
        return 1.0
    elif unit == "mparams":
        return 1.0e6
    else:
        raise ValueError(f'Unknown params unit "{unit}"')


def calc_params(
    m: torch.nn.Module, unit: str = "Mparams", only_trainable: bool = False
) -> float:
    # https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model

    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    res = sum(p.numel() for p in unique)
    divisor = _get_params_divisor(unit)
    return res / divisor


def _calc_node_params(
    node: torch.fx.Node,
    module_dict: dict[str, torch.nn.Module],
    unit: str,
    only_trainable: bool,
) -> float:
    assert node.op == "call_module" and isinstance(node.target, str)
    module = module_dict[node.target]
    return calc_params(module, unit, only_trainable)


def calc_params_split(
    m: torch.nn.Module,
    *,
    unit: str = "Mparams",
    only_trainable: bool = False,
    split_fn: collections.abc.Callable[
        [torch.fx.Node, dict[str, torch.nn.Module]], str
    ] = core.get_fxnode_name,
    sort_by: str = "params_r",
) -> dict[str, tuple[float, int]]:
    res: dict[str, tuple[float, int]] = {}
    sort_by_to_key_fn = {
        "params": lambda key: res[key][0],
        "count": lambda key: res[key][1],
        "params_r": lambda key: -res[key][0],
        "count_r": lambda key: -res[key][1],
    }

    gm = core.symbolic_trace_if_needed(m)

    module_dict = dict(gm.named_modules())
    for node in gm.graph.nodes:
        if fxfilters.is_with_params(node, module_dict):
            key = split_fn(node, module_dict)
            num_params = _calc_node_params(node, module_dict, unit, only_trainable)
            if key in res:
                old_num_params, old_count = res[key]
                res[key] = (old_num_params + num_params, old_count + 1)
            else:
                res[key] = (num_params, 1)
    return {key: res[key] for key in sorted(res.keys(), key=sort_by_to_key_fn[sort_by])}
