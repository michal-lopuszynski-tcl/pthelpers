import functools
from typing import Union

import torch.fx

__all__ = (
    "get_first_fxnode",
    "get_fxnode",
    "get_fxnode_name",
    "get_fxnode_target_name",
    "get_last_fxnode",
    "get_module",
    "make_rand_tensors",
    "replace_fxsubmodule",
    "symbolic_trace_if_needed",
    "get_devices",
    "get_device",
)


def symbolic_trace_if_needed(m: torch.nn.Module) -> torch.fx.GraphModule:
    if isinstance(m, torch.fx.GraphModule):
        return m
    else:
        return torch.fx.symbolic_trace(m)


def get_module(module: torch.nn.Module, target: str) -> torch.nn.Module:
    names = target.split(sep=".")
    return functools.reduce(getattr, names, module)


def get_fxnode(m: torch.fx.GraphModule, item: Union[str, int]) -> torch.fx.node.Node:
    if isinstance(item, str):
        for n in m.graph.nodes:
            if n.name == item:
                return n
        raise KeyError(f'Item "{item}" not found in graph {m}')
    elif isinstance(item, int):
        nodes = list(m.graph.nodes)
        return nodes[item]
    else:
        raise TypeError(f"Unsuported index type {type(item)}")


def get_first_fxnode(m: torch.fx.GraphModule) -> torch.fx.node.Node:
    return next(iter(m.graph.nodes))


def get_last_fxnode(m: torch.fx.GraphModule) -> torch.fx.node.Node:
    return next(iter(reversed(m.graph.nodes)))


def get_fxnode_target_name(
    node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]
) -> str:
    if node.op == "call_function":
        assert not isinstance(node.target, str)
        return node.target.__name__
    elif node.op == "call_method":
        assert isinstance(node.target, str)
        return node.target
    elif node.op == "call_module":
        assert isinstance(node.target, str)
        t = module_dict[node.target]
        # return f"{type(t).__module__}.{type(t).__name__}"
        return f"{node.target}\n{type(t).__name__}"
    elif node.op in ["output", "placeholder", "get_attr"]:
        return ""
    else:
        assert False, f"Unknown op type {node.op}"


def get_fxnode_name(
    node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]
) -> str:
    node_op_to_abbreviation = {
        "placeholder": "pla",
        "get_attr": "att",
        "call_function": "fun",
        "call_method": "met",
        "call_module": "mod",
        "output": "out",
    }

    if node.op == "output":
        return "out.out"
    elif node.op == "placeholder":
        return "pla.pla"
    else:
        op = node_op_to_abbreviation[node.op]
        op_name = get_fxnode_target_name(node, module_dict)
        return op + "." + op_name


def _split_parent_child_name(target: str) -> tuple[str, str]:
    *parent, name = target.rsplit(".", 1)
    return parent[0] if parent else "", name


def replace_fxsubmodule(
    m: torch.fx.GraphModule, node: torch.fx.Node, new_module: torch.nn.Module
) -> None:
    if node.op != "call_module":
        raise ValueError("Expected call_module node, got {node.op} for {node.name}")
    parent_name, name = _split_parent_child_name(str(node.target))
    modules = dict(m.named_modules())
    setattr(modules[parent_name], name, new_module)


def make_rand_tensors(
    shapes: tuple[tuple[int, ...]], device: torch.device
) -> tuple[torch.Tensor, ...]:
    return tuple(torch.normal(0.0, 1.0, size=shape).to(device) for shape in shapes)


def get_devices(m: torch.nn.Module) -> list[torch.device]:
    devices = []
    for p in m.parameters():
        if p.device not in devices:
            devices.append(p.device)
    return devices


def get_device(m: torch.nn.Module) -> torch.device:
    p = next(m.parameters(), None)
    if p is None:
        return torch.device("cpu")
    else:
        return p.device
