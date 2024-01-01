import copy
import logging
import math
from collections.abc import Callable
from types import ModuleType
from typing import Any, Optional, Union

import torch

from .. import core

__all__ = [
    "PartialTracer",
    "symbolic_trace_nested",
    "symbolic_trace_one_level",
    "symbolic_trace_partial",
    "symbolic_trace_tolerant",
    "find_untraceable_nodes",
    "is_leaf_module_default",
    "is_leaf_module_always",
]


logger = logging.getLogger(__name__)


class PartialTracer(torch.fx.Tracer):
    def __init__(
        self,
        is_leaf_fn: Callable[[torch.nn.Module, str], bool],
        autowrap_modules: tuple[ModuleType] = (math,),
        autowrap_functions: tuple[Callable, ...] = (),
        param_shapes_constant: bool = False,
    ):
        super().__init__(
            autowrap_modules=autowrap_modules,
            autowrap_functions=autowrap_functions,
            param_shapes_constant=param_shapes_constant,
        )
        self.is_leaf_fn = is_leaf_fn

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        return self.is_leaf_fn(m, module_qualified_name)


def is_leaf_module_default(m: torch.nn.Module, module_qualified_name: str) -> bool:
    "Default torch.fx implementation"
    return (
        m.__module__.startswith("torch.nn") or m.__module__.startswith("torch.ao.nn")
    ) and not isinstance(m, torch.nn.Sequential)


def is_leaf_module_always(m: torch.nn.Module, module_qualified_name: str) -> bool:
    return True


def symbolic_trace_partial(
    root: Union[torch.nn.Module, Callable[..., Any]],
    is_leaf_fn: Callable[[torch.nn.Module, str], bool],
    concrete_args: Optional[dict[str, Any]] = None,
) -> torch.fx.GraphModule:
    tracer = PartialTracer(is_leaf_fn)
    graph = tracer.trace(root, concrete_args)
    name = (
        root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    )
    return torch.fx.GraphModule(tracer.root, graph, name)


def symbolic_trace_one_level(
    root: Union[torch.nn.Module, Callable[..., Any]],
    concrete_args: Optional[dict[str, Any]] = None,
) -> torch.fx.GraphModule:
    return symbolic_trace_partial(
        root, is_leaf_module_always, concrete_args=concrete_args
    )


def _symbolic_trace_nested(
    root: Union[torch.nn.Module, Callable[..., Any]],
    concrete_args: Optional[dict[str, Any]] = None,
    warn_untraceable: bool = True,
) -> torch.fx.GraphModule:
    gm = symbolic_trace_one_level(root, concrete_args=concrete_args)

    for n in gm.graph.nodes:
        if n.op == "call_module":
            assert isinstance(n.target, str)
            m = core.get_module(gm, n.target)
            if not isinstance(m, torch.fx.GraphModule) and not is_leaf_module_default(
                m, n.target
            ):
                tr_m = None
                try:
                    tr_m = _symbolic_trace_nested(m, warn_untraceable=warn_untraceable)
                except Exception:
                    if warn_untraceable:
                        logger.warn(f"Failed to trace {n.name}, {m}")
                if tr_m is not None:
                    core.replace_fxsubmodule(gm, n, tr_m)
    return gm


def symbolic_trace_nested(
    root: Union[torch.nn.Module, Callable[..., Any]],
    concrete_args: Optional[dict[str, Any]] = None,
    warn_untraceable: bool = True,
) -> torch.fx.GraphModule:
    # To avoid interfering with the original module
    root_copy = copy.deepcopy(root)
    return _symbolic_trace_nested(
        root_copy, concrete_args=concrete_args, warn_untraceable=warn_untraceable
    )


def symbolic_trace_tolerant(
    root: Union[torch.nn.Module, Callable[..., Any]],
    concrete_args: Optional[dict[str, Any]] = None,
    warn_untraceable: bool = False,
) -> torch.fx.GraphModule:
    def _is_leaf_module(m: torch.nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, torch.fx.GraphModule):
            return False
        return True

    gm_tmp = symbolic_trace_nested(
        root, concrete_args=concrete_args, warn_untraceable=warn_untraceable
    )
    return symbolic_trace_partial(gm_tmp, _is_leaf_module)


def find_untraceable_nodes(
    gm: torch.fx.GraphModule,
) -> tuple[list[torch.fx.Node], list[int]]:
    nodes = []
    indices = []

    for i, node in enumerate(gm.graph.nodes):
        if node.op == "call_module":
            m = core.get_module(gm, node.target)
            if not is_leaf_module_default(m, node.target):
                indices.append(i)
                nodes.append(node)

    return nodes, indices
