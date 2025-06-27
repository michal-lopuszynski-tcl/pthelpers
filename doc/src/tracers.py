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
# # Tracing examples

# %% [markdown]
# We are using the following versions:

# %%
# | echo: false
print(get_versions())


# %% [markdown]
# Let us explore the `NAFNet` model for super resolution and denoising:
#
# * paper: [arXiv:2204.04676](https://arxiv.org/abs/2204.04676)
#
# * code: [https://github.com/megvii-research/NAFNet](https://github.com/megvii-research/NAFNet)

# %%
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
# The function `symbolic_trace_leaf_types` allows us for easy blacklisting certain modules from tracing.
#
# This can make e.g.  visualizations more helpful.

# %%
NAFNET_LEAF_MODULE_TYPES = (
    nafnet_arch.LayerNorm2d,
    nafnet_arch.NAFBlock,
    nafnet_arch.SimpleGate,
)

model_fx = pthelpers.fxtracers.symbolic_trace_leaf_types(
    model,
    leaf_module_types=NAFNET_LEAF_MODULE_TYPES,
)

pthelpers.vis.vis_module(model_fx)

# %% [markdown]
# We can also have a look at the individual `NAFBlock`.

# %%
model.encoders[0][0]

# %%
block_fx = pthelpers.fxtracers.symbolic_trace_leaf_types(
    model.encoders[0][0],
    leaf_module_types=NAFNET_LEAF_MODULE_TYPES,
)

pthelpers.vis.vis_module(block_fx)
