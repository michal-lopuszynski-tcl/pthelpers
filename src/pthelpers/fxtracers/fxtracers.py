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
    "symbolic_trace_leaf_fn",
    "symbolic_trace_leaf_types",
    "symbolic_trace_tolerant",
    "find_untraceable_nodes",
    "is_leaf_module_default",
]


logger = logging.getLogger(__name__)


class PartialTracer(torch.fx.Tracer):
    def __init__(
        self,
        is_leaf_module_fn: Callable[[torch.nn.Module, str], bool],
        autowrap_modules: tuple[ModuleType] = (math,),
        autowrap_functions: tuple[Callable, ...] = (),
        param_shapes_constant: bool = False,
    ):
        super().__init__(
            autowrap_modules=autowrap_modules,
            autowrap_functions=autowrap_functions,
            param_shapes_constant=param_shapes_constant,
        )
        self.is_leaf_fn = is_leaf_module_fn

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        return self.is_leaf_fn(m, module_qualified_name)


def is_leaf_module_default(m: torch.nn.Module, module_qualified_name: str) -> bool:
    "Default torch.fx implementation"
    return (
        m.__module__.startswith("torch.nn") or m.__module__.startswith("torch.ao.nn")
    ) and not isinstance(m, torch.nn.Sequential)


# def is_leaf_module_always(m: torch.nn.Module, module_qualified_name: str) -> bool:
#     return True


def symbolic_trace_leaf_fn(
    root: Union[torch.nn.Module, Callable[..., Any]],
    is_leaf_module_fn: Optional[Callable[[torch.nn.Module, str], bool]] = None,
    concrete_args: Optional[dict[str, Any]] = None,
) -> torch.fx.GraphModule:
    """
    Exactly like torch.fx.symbolic_trace, but lets you specify which modules are leafs
    by functional interface.

    Convenience wrapper for creating custom tracer class.
    """
    tracer = PartialTracer(is_leaf_module_fn)
    graph = tracer.trace(root, concrete_args)
    name = (
        root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    )
    return torch.fx.GraphModule(tracer.root, graph, name)


def symbolic_trace_leaf_types(
    root: Union[torch.nn.Module, Callable[..., Any]],
    leaf_module_types: tuple[type, ...],
    concrete_args: Optional[dict[str, Any]] = None,
) -> torch.fx.GraphModule:

    def __is_leaf_module(m: torch.nn.Module, module_qualified_name: str):
        # Why `type(m) is t` instead of `isinstance(m, non_leaf_module_types`)?
        # Some libraries inherit from say torch.nn.Linear, in such a case we want to be
        # explicit and let the user decide what types exactly should be traced
        tm = type(m)
        for t in leaf_module_types:
            if tm is t:
                return True
        return is_leaf_module_default(m, module_qualified_name)

    return symbolic_trace_leaf_fn(root, __is_leaf_module, concrete_args)


def symbolic_trace_one_level(
    root: Union[torch.nn.Module, Callable[..., Any]],
    concrete_args: Optional[dict[str, Any]] = None,
    non_leaf_module_types: tuple[type, ...] = (torch.nn.Sequential,),
) -> torch.fx.GraphModule:

    def __is_leaf_module(m: torch.nn.Module, module_qualified_name: str) -> bool:
        # Why `type(m) is t` instead of `isinstance(m, non_leaf_module_types`)?
        # Some libraries inherit from say torch.nn.Linear, in such a case we want to be
        # explicit and let the user decide what types exactly should be traced
        tm = type(m)
        for t in non_leaf_module_types:
            if tm is t:
                return False
        return True

    return symbolic_trace_leaf_fn(root, __is_leaf_module, concrete_args=concrete_args)


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
                        logger.warning(f"Failed to trace {n.name}, {m}")
                if tr_m is not None:
                    core.replace_fxsubmodule(gm, n, tr_m)
    return gm


def symbolic_trace_nested(
    root: Union[torch.nn.Module, Callable[..., Any]],
    concrete_args: Optional[dict[str, Any]] = None,
    warn_untraceable: bool = True,
) -> torch.fx.GraphModule:
    return _symbolic_trace_nested(
        root, concrete_args=concrete_args, warn_untraceable=warn_untraceable
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
    return symbolic_trace_leaf_fn(gm_tmp, _is_leaf_module)


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


# def _is_compound_module(m: torch.nn.Module) -> bool:
#     return len(list(m.children())) > 0


# def _get_children_module_names(m: torch.nn.Module) -> list[str]:
#     return [n for n, _ in m.named_children()]


# def _symbolic_trace(
#     root: Union[torch.nn.Module, Callable[..., Any]],
#     leaf_module_types: tuple[type, ...] = (),
# ) -> torch.fx.GraphModule:
#     def __is_leaf_module(m: torch.nn.Module, module_qualified_name: str) -> bool:
#         if isinstance(m, leaf_module_types):
#             return True
#         return is_leaf_module_default(m, module_qualified_name)

#     root_traced = symbolic_trace_partial(root, __is_leaf_module)
#     return root_traced


# def _trace_nested_modules_in_place(
#     root: torch.nn.Module,
#     *,
#     leaf_module_types: tuple[type, ...],
#     nest_module_types: tuple[type, ...],
#     module_path: tuple[str, ...] = (),
# ) -> None:

#     children_module_names = _get_children_module_names(root)

#     for child_module_name in children_module_names:
#         child_module_path = ".".join((*module_path, child_module_name))
#         logger.info(f"Analyzing {child_module_path}")
#         module = root.get_submodule(child_module_name)
#         if _is_compound_module(module):
#             _trace_nested_modules_in_place(
#                 module,
#                 leaf_module_types=leaf_module_types,
#                 nest_module_types=nest_module_types,
#                 module_path=(*module_path, child_module_name),
#             )
#         if isinstance(module, nest_module_types):
#             child_module_path = ".".join((*module_path, child_module_name))
#             module_type_name = core.get_type_name(module)
#             ms = f"Replacing submodule {child_module_path} of type {module_type_name}"
#             logger.info(ms)
#             module_tr = _symbolic_trace(module, leaf_module_types=leaf_module_types)
#             core.replace_submodule_in_place(root, child_module_name, module_tr)


# def symbolic_trace_easy(
#     root: Union[torch.nn.Module, Callable[..., Any]],
#     *,
#     leaf_module_types: tuple[type, ...] = (),
#     nest_module_types: tuple[type, ...] = (),
# ) -> torch.fx.GraphModule:

#     # WARNING! Currently, this is desrtuctive to root (original module)

#     if isinstance(root, torch.nn.Module) and nest_module_types:
#         leaf_module_types = (*leaf_module_types, torch.fx.GraphModule)
#         _trace_nested_modules_in_place(
#             root=root,
#             nest_module_types=nest_module_types,
#             leaf_module_types=leaf_module_types,
#         )
#     return _symbolic_trace(root, leaf_module_types=leaf_module_types)
