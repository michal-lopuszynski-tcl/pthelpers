import collections.abc
import os
import random
import re
import string
from typing import Any, Dict, List, Optional, Union

import graphviz  # type: ignore
import torch

from .. import core, fxfilters
from . import styles


class Digraph(graphviz.graphs.Digraph):
    """
    This class circumvents the problem of coliding styles in svg figures
    visualized in jupyter.

    The solution is a bit hacky, and it might break or become obsolete in future
    versions ofgraphviz jupyter.

    It works by addign random prefixes by svg styles.
    """

    @staticmethod
    def _get_random_string(n: int) -> str:
        return "".join(random.choice(string.ascii_lowercase) for _ in range(n))

    @staticmethod
    def _randomize_svg_style(s: str) -> str:
        style_id_regexp_str = '"l_[^"]*"'
        # print(a)
        style_id_regexp = re.compile(style_id_regexp_str)
        styles = re.findall(style_id_regexp, s)
        for style in styles:
            random_prefix = Digraph._get_random_string(6)
            sss = f"id={style}"
            ttt = 'id="' + random_prefix + "_" + style[1:-1] + '"'
            # print(sss, ttt)
            s = s.replace(sss, ttt)
            sss = "(#" + style[1:-1] + ")"
            ttt = "(#" + random_prefix + "_" + style[1:-1] + ")"
            s = s.replace(sss, ttt)
        return s

    def _repr_image_svg_xml(self) -> str:
        repr = super()._repr_image_svg_xml()
        return self._randomize_svg_style(repr)


def _add_edges_in_place(edges: list[tuple[str, str]], cur_node: torch.fx.Node) -> None:
    for a in cur_node.all_input_nodes:
        edges.append((a.name, cur_node.name))


def _add_node_to_graph(
    dot: graphviz.graphs.Digraph, node_name: str, node_style: dict[str, str]
) -> None:
    dot.node(node_name, **node_style)


def _add_edge_to_graph(
    dot: graphviz.graphs.Digraph,
    node_name1: str,
    node_name2: str,
    edge_style: dict[str, str],
) -> None:
    dot.edge(node_name1, node_name2, **edge_style)


def vis_module(
    module: torch.nn.Module,
    input_shapes: Optional[tuple[tuple[int, ...]]] = None,
    ignore_getattr: bool = False,
    get_style_fn: collections.abc.Callable[..., Any] = styles.get_std_style,
    saving_path: Optional[Union[str, bytes, os.PathLike]] = None,
    saving_format: str = "pdf",
) -> graphviz.Digraph:
    traced_module = core.symbolic_trace_if_needed(module)

    if input_shapes is not None:
        compute_result = True
        sample_input_tensors = (torch.rand(*shape) for shape in input_shapes)
        args_iter = iter(sample_input_tensors)
    else:
        compute_result = False
        args_iter = None

    env: Dict[str, Dict[str, Any]] = {}
    output_node_names: List[str] = []
    module_dict = dict(traced_module.named_modules())

    edges: list[tuple[str, str]] = []

    def _load_arg(
        a: Union[tuple[torch.fx.node.Argument], dict[str, torch.fx.node.Argument]]
    ) -> Any:
        return torch.fx.graph.map_arg(a, lambda n: env[n.name]["node_result"])

    def _fetch_attr(target: str) -> Any:
        target_atoms = target.split(".")
        attr_itr = traced_module
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(
                    f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}"
                )
            attr_itr = getattr(attr_itr, atom)
        return attr_itr

    result = None  # To make linter happy

    for i, node in enumerate(traced_module.graph.nodes):
        node_meta = {"node": node, "node_idx": i}

        if node.op == "placeholder":
            if compute_result:
                assert args_iter is not None
                result = next(args_iter)
        elif node.op == "get_attr":
            if compute_result:
                result = _fetch_attr(node.target)
        elif node.op == "call_function":
            if compute_result:
                result = node.target(*_load_arg(node.args), **_load_arg(node.kwargs))
        elif node.op == "call_method":
            if compute_result:
                self_obj, *sample_input_tensors = _load_arg(node.args)
                kwargs = _load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*sample_input_tensors, **kwargs)
        elif node.op == "call_module":
            if compute_result:
                result = module_dict[node.target](
                    *_load_arg(node.args), **_load_arg(node.kwargs)
                )
        elif node.op == "output":
            output_node_names.append(node.name)
        else:
            assert False, f"Unknown op type {node.op}"

        _add_edges_in_place(edges, node)
        if isinstance(result, torch.Tensor):
            node.shape = result.shape
            node.dtype = result.dtype

        node_meta["node_result"] = result
        env[node.name] = node_meta

    graph_style = get_style_fn(element="graph")
    dot = graphviz.Digraph(**graph_style)

    for node_name, node_meta in env.items():
        if ignore_getattr and fxfilters.is_get_attr(node_meta["node"], module_dict):
            continue
        node_style = get_style_fn(
            element="node", node_meta1=node_meta, module_dict=module_dict
        )
        _add_node_to_graph(dot, node_name, node_style)

    for node_name1, node_name2 in edges:
        node_meta1, node_meta2 = env[node_name1], env[node_name2]
        if ignore_getattr and (
            fxfilters.is_get_attr(node_meta1["node"], module_dict)
            or fxfilters.is_get_attr(node_meta2["node"], module_dict)
        ):
            continue
        edge_style = get_style_fn(
            element="edge",
            node_meta1=node_meta1,
            node_meta2=node_meta2,
            module_dict=module_dict,
        )
        _add_edge_to_graph(dot, node_name1, node_name2, edge_style)

    if saving_path is not None:
        dot.format = saving_format
        dot.render(saving_path)
    dot.__class__ = Digraph
    return dot
