# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
#| echo: false
# %load_ext autoreload
# %autoreload 2

# %%
# | echo: false
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def setup_logging():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s.%(msecs)03d500: %(levelname).1s %(name)s.py:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Here you put modules where you want more verbose logging

    for module_name in [__name__]:
        logging.getLogger(module_name).setLevel(logging.INFO)


setup_logging()


# %%
# | echo: false
def get_versions():
    import torch
    import pthelpers

    return f"torch version     = {torch.__version__}\npthelpers_version = {pthelpers.__version__}"


# %% [markdown]
# ## Advanced visualzation

# %% [markdown]
# We are using the following versions:

# %%
# | echo: false
print(get_versions())

# %% [markdown]
# Let us create our test model:

# %%
import torch
import pthelpers
import nafnet_arch

img_channel = 3
width = 16
enc_blks = [1, 1, 1, 1]
middle_blk_num = 1
dec_blks = [1, 1, 1, 1]

model = nafnet_arch.NAFNet(
    img_channel=img_channel,
    width=width,
    middle_blk_num=middle_blk_num,
    enc_blk_nums=enc_blks,
    dec_blk_nums=dec_blks,
)
model.eval()


# %% [markdown]
# Create style coloring by number of params:

# %%
def _get_num_params(module):
    return sum(dict((p.data_ptr(), p.numel()) for p in module.parameters()).values())


def get_max_params(m):
    m_traced = torch.fx.symbolic_trace(m)
    module_dict = dict(m_traced.named_modules())
    max_num_params = -1
    for node in m_traced.graph.nodes:
        if node.op == "call_module":
            module = module_dict[node.target]
            num_params = _get_num_params(module)
            if num_params > max_num_params:
                max_num_params = num_params
    return max_num_params


def _get_color(x):
    x = max(min(1.0, x), 0.0)
    if x > 1.0e-6:
        res = "#"
        min_color = [255.0, 230.0, 230.0]
        max_color = [255.0, 0.0, 0.0]
        for minc, maxc in zip(min_color, max_color):
            c = round(minc + x * (maxc - minc))
            res += f"{c:02X}"
        return res
    else:
        return "#ffffff"


def make_num_params_style_mini_fn(scale):
    def __get_style(*, element, node_meta1=None, node_meta2=None, module_dict=None):
        style = pthelpers.vis.get_std_min_style(
            element=element,
            node_meta1=node_meta1,
            node_meta2=node_meta1,
            module_dict=module_dict,
        )
        if element == "node":
            node = node_meta1["node"]
            if node.op == "call_module":
                module = module_dict[node.target]
                num_params = _get_num_params(module)
                old_tooltip = style["tooltip"]
                style["tooltip"] = f"#p={num_params}\n\n" + old_tooltip
                style["fillcolor"] = _get_color(num_params / scale)
            else:
                old_tooltip = style["tooltip"]
                style["tooltip"] = "#p=0\n\n" + old_tooltip
                style["fillcolor"] = _get_color(0.0)
        return style

    return __get_style


def make_num_params_style_fn(scale):
    def __get_style(*, element, node_meta1=None, node_meta2=None, module_dict=None):
        style = pthelpers.vis.get_std_style(
            element=element,
            node_meta1=node_meta1,
            node_meta2=node_meta1,
            module_dict=module_dict,
        )
        if element == "node":
            node = node_meta1["node"]
            if node.op == "call_module":
                module = module_dict[node.target]
                num_params = _get_num_params(module)
                style["fillcolor"] = _get_color(num_params / scale)
            else:
                style["fillcolor"] = _get_color(0.0)
        return style

    return __get_style


# %%
NAFNET_LEAF_MODULE_TYPES = (
    nafnet_arch.NAFBlock,
    nafnet_arch.SimpleGate,
)
NAFNET_TOT_PARAMS = get_max_params(model)

model_fx = pthelpers.fxtracers.symbolic_trace_leaf_types(
    model,
    leaf_module_types=NAFNET_LEAF_MODULE_TYPES,
)

pthelpers.vis.vis_module(
    model_fx,
    input_shapes=(1,3,256,256),
    get_style_fn=make_num_params_style_fn(NAFNET_TOT_PARAMS),
)

# %%
pthelpers.vis.vis_module(model_fx, input_shapes=(1, 3, 256, 256), get_style_fn=make_num_params_style_mini_fn(NAFNET_TOT_PARAMS))


# %%
block_fx = pthelpers.fxtracers.symbolic_trace_leaf_types(
    model.encoders[3][0],
    leaf_module_types=NAFNET_LEAF_MODULE_TYPES,
)

BLOCK_TOT_PARAMS = get_max_params(block_fx)

pthelpers.vis.vis_module(block_fx, input_shapes=(1, 128, 32, 32), get_style_fn=make_num_params_style_fn(BLOCK_TOT_PARAMS))

# %%
pthelpers.vis.vis_module(block_fx, input_shapes=(1, 128, 32, 32), get_style_fn=make_num_params_style_mini_fn(BLOCK_TOT_PARAMS))
