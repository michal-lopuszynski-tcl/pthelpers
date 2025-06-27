# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
"""

import torch


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.shape
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(torch.nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter("weight", torch.nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", torch.nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class DyT(torch.nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = torch.nn.Parameter(torch.ones(num_features))
        self.bias = torch.nn.Parameter(torch.rand(num_features))
        self.tanh = torch.torch.nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.alpha * x)
        return x * self.weight + self.bias


class SimpleGate(torch.nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(torch.nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = torch.nn.Conv2d(
            in_channels=c,
            out_channels=dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=True,
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=dw_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        # Simplified Channel Attention
        self.sca = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(
                in_channels=dw_channel // 2,
                out_channels=dw_channel // 2,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            ),
        )

        # SimpleGate
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = torch.nn.Conv2d(
            in_channels=c,
            out_channels=ffn_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv5 = torch.nn.Conv2d(
            in_channels=ffn_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.norm1 = torch.nn.LayerNorm(c)
        self.norm2 = torch.nn.LayerNorm(c)

        self.dropout1 = (
            torch.nn.Dropout(drop_out_rate)
            if drop_out_rate > 0.0
            else torch.nn.Identity()
        )
        self.dropout2 = (
            torch.nn.Dropout(drop_out_rate)
            if drop_out_rate > 0.0
            else torch.nn.Identity()
        )

        self.beta = torch.nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = torch.nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta
        x = y.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(torch.nn.Module):
    def __init__(
        self,
        img_channel=3,
        width=16,
        middle_blk_num=1,
        enc_blk_nums=[],
        dec_blk_nums=[],
    ):
        super().__init__()

        self.intro = torch.nn.Conv2d(
            in_channels=img_channel,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )
        self.ending = torch.nn.Conv2d(
            in_channels=width,
            out_channels=img_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

        self.encoders = torch.nn.ModuleList()
        self.decoders = torch.nn.ModuleList()
        self.middle_blks = torch.nn.ModuleList()
        self.ups = torch.nn.ModuleList()
        self.downs = torch.nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                torch.nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
            self.downs.append(torch.nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = torch.nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(chan, chan * 2, 1, bias=False),
                    torch.nn.PixelShuffle(2),
                )
            )
            chan = chan // 2
            self.decoders.append(
                torch.nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return None, x


# if __name__ == "__main__":
#
#     img_channel = 3
#     width = 16
#
#     # enc_blks = [2, 2, 4, 8]
#     # middle_blk_num = 12
#     # dec_blks = [2, 2, 2, 2]
#
#     enc_blks = [1, 1, 1, 1]
#     middle_blk_num = 1
#     dec_blks = [1, 1, 1, 1]
#
#     model = NAFNet(
#         img_channel=img_channel,
#         width=width,
#         middle_blk_num=middle_blk_num,
#         enc_blk_nums=enc_blks,
#         dec_blk_nums=dec_blks,
#     )
#     model.eval()
#
#     from ptflops import get_model_complexity_info
#
#     inp_shape = (3, 256, 256)
#     # inp_shape = (3, 1920, 1088)
#     macs, params = get_model_complexity_info(
#         model, inp_shape, verbose=False, print_per_layer_stat=False
#     )
#     print("NAFNet:", macs, params)
