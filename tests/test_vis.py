import pytest
import torch

import pthelpers.vis


class SampleNet1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=3, out_channels=8, kernel_size=3, padding="same"
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=3, out_channels=8, kernel_size=3, padding="same"
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=3, out_channels=8, kernel_size=3, padding="same"
        )
        self.conv4 = torch.nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, padding="same"
        )
        self.conv5 = torch.nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding="same"
        )
        self.conv6 = torch.nn.Conv2d(
            in_channels=24, out_channels=32, kernel_size=3, padding="same"
        )
        self.conv7 = torch.nn.Conv2d(
            in_channels=8, out_channels=32, kernel_size=3, padding="same"
        )
        self.conv8 = torch.nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding="same"
        )
        self.logits = torch.nn.Parameter(torch.empty((1, 10, 1, 1)))
        torch.nn.init.constant_(self.logits, 1.0)

    def forward(self, x):
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x)
        x_conv3 = self.conv3(x)
        x_act_conv1 = torch.nn.functional.relu(x_conv1)
        x_conv4 = self.conv4(x_act_conv1) + self.logits.mean()
        x_act_conv4 = torch.nn.functional.relu(x_conv4)
        x_conv5 = self.conv5(x_act_conv4)
        x_act_conv2 = torch.nn.functional.relu(x_conv2)
        x_act1_conv3 = torch.nn.functional.relu(x_conv3)
        x_act2_conv3 = torch.sigmoid(x_conv3)
        x_conv7 = self.conv7(x_act2_conv3)
        x_concat1 = torch.cat([x_act_conv2, x_act1_conv3, x_act2_conv3], dim=1)
        x_conv6 = self.conv6(x_concat1)
        x_concat2 = torch.cat([x_act_conv1, x_act_conv2], dim=1)
        x_conv8 = self.conv8(x_concat2)
        x_sum1 = x_conv5 + x_conv8
        return (
            x_sum1.mean()
            + x_conv5.mean()
            + x_conv6.mean()
            + x_conv7.mean()
            + self.logits.mean()
        )


def make_arg_list():
    input_shapes = [(), None, [(1, 3, 224, 224)]]
    ignore_getattrs = [None, True, False]
    get_style_fns = [
        None,
        pthelpers.vis.get_raw_style,
        pthelpers.vis.get_std_style,
        pthelpers.vis.get_std_min_style,
    ]
    saves = [True, False]
    saving_formats = ["pdf", "svg"]
    i = 1
    res = []
    for input_shape in input_shapes:
        for ignore_getattr in ignore_getattrs:
            for get_style_fn in get_style_fns:
                for save in saves:
                    for saving_format in saving_formats:
                        res.append(
                            [
                                str(i),
                                input_shape,
                                ignore_getattr,
                                get_style_fn,
                                save,
                                saving_format,
                            ]
                        )
    return res


def verify_vis_module(
    tmp_path,
    module_factory,
    name,
    input_shapes,
    ignore_getattr,
    get_style_fn,
    save,
    saving_format,
):
    nn = module_factory()
    kwargs = {}

    if input_shapes is None or len(input_shapes) != 0:
        kwargs["input_shapes"] = input_shapes

    if ignore_getattr is not None:
        kwargs["ignore_getattr"] = ignore_getattr

    if get_style_fn is not None:
        kwargs["get_style_fn"] = get_style_fn

    if saving_format is not None:
        kwargs["saving_format"] = saving_format
    else:
        saving_format = "pdf"

    if save is not None:
        kwargs["saving_path"] = tmp_path / name
        saving_path_dot = tmp_path / name
        saving_path_format = tmp_path / (name + "." + saving_format)

    dot = pthelpers.vis.vis_module(nn, **kwargs)

    assert dot is not None

    if save is not None:
        assert saving_path_dot.is_file()
        assert saving_path_format.is_file()


@pytest.mark.parametrize(
    "name, input_shapes, ignore_getattr, get_style_fn, save, saving_format",
    make_arg_list()[-2:],
)
def test_vis_module1(
    tmp_path, name, input_shapes, ignore_getattr, get_style_fn, save, saving_format
):
    verify_vis_module(
        tmp_path,
        SampleNet1,
        name,
        input_shapes,
        ignore_getattr,
        get_style_fn,
        save,
        saving_format,
    )
