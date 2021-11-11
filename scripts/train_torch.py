import numpy as np
import torch

from stardist.models import Config2D, StarDist2D
from csbdeep.utils.tf import keras_import

from backbones import Unet2d


def circle_image(shape=(128, 128), radius=None, center=None, eps=None):
    if center is None:
        center = (0,) * len(shape)
    if radius is None:
        radius = min(shape) // 4
    if eps is None:
        eps = (1,) * len(shape)
    assert len(shape) == len(eps)
    xs = tuple(np.arange(s) - s // 2 for s in shape)
    Xs = np.meshgrid(*xs, indexing="ij")
    R = np.sqrt(np.sum([(X - c) ** 2 / _eps**2 for X, c,
                        _eps in zip(Xs, center, eps)], axis=0))
    img = (R < radius).astype(np.uint16)
    return img


class StarDistTorch2d(torch.nn.Module):

    def __init__(self, conf, name, basedir):

        super().__init__()

        self.conf = conf

        self.encoder = Unet2d(
            in_channels=conf.n_channel_in,
            # TODO adapt
            initial_fmaps=64,
            fmap_inc_factor=2,
            downsample_factors=((4, 4), (2, 2), (2, 2)),
            out_channels=16,
            batch_norm=False,
            padding=1,
            padding_mode='replicate',
        )

        self.head = torch.nn.Conv2d(
            in_channels=16,
            out_channels=1 + conf.n_rays,
            kernel_size=1,
        )

    def forward(self, x):

        z = self.encoder(x)
        out = self.head(z)

        return out


tmpdir = "./tmp"
n_rays = 17
n_channel = 1

img = circle_image(shape=(160, 160))
imgs = np.repeat(img[np.newaxis], 3, axis=0)

imgs = np.repeat(imgs[..., np.newaxis], n_channel, axis=-1)

X = imgs + .6 * np.random.uniform(0, 1, imgs.shape)
Y = (imgs if imgs.ndim == 3 else imgs[..., 0]).astype(int)

X = np.moveaxis(X, -1, 1)

X = torch.as_tensor(X, dtype=torch.float)
Y = torch.as_tensor(Y)

data = torch.utils.data.TensorDataset(X, Y)

dataloader = torch.utils.data.DataLoader(
    data,
    batch_size=2,
)

conf = Config2D(
    n_rays=n_rays,
    n_channel_in=n_channel,
    use_gpu=False,
    train_epochs=2,
    train_steps_per_epoch=1,
    train_batch_size=2,
    # TODO adapt patch size to model
    train_patch_size=(128, 128),
)
model = StarDistTorch2d(conf, name='stardist', basedir=str(tmpdir))

for batch in dataloader:
    x, y = batch

    out = model(x)


# model = StarDist2D(conf, name='stardist', basedir=str(tmpdir))
# model.train(X, Y, validation_data=(X[:2], Y[:2]), workers=workers)
