import numpy as np
import torch
import augmend

from stardist.models import Config2D, StarDist2D, StarDistData2D
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

        # TODO can I (ab)use stardist tb logging?
        super().__init__()

        self.conf = conf

        num_embs = 32
        self.encoder = Unet2d(
            in_channels=conf.n_channel_in,
            # initial_fmaps=64,
            initial_fmaps=8,
            fmap_inc_factor=2,
            # downsample_factors=((4, 4), (2, 2), (2, 2)),
            downsample_factors=((2, 2), (2, 2), (2, 2)),
            out_channels=num_embs,
            batch_norm=False,
            padding=1,
            padding_mode='replicate',
        )

        self.head_prob = torch.nn.Conv2d(
            in_channels=num_embs,
            out_channels=1,
            kernel_size=1,
        )

        self.head_dist = torch.nn.Conv2d(
            in_channels=num_embs,
            out_channels=conf.n_rays,
            kernel_size=1,
        )

    def forward(self, x):

        z = self.encoder(x)
        prob = self.head_prob(z)
        dist = self.head_dist(z)

        return prob, dist


tmpdir = "./tmp"
n_rays = 17
n_channel = 1

img = circle_image(shape=(160, 160))
imgs = np.repeat(img[np.newaxis], 20, axis=0)

imgs = np.repeat(imgs[..., np.newaxis], n_channel, axis=-1)

X = imgs + .6 * np.random.uniform(0, 1, imgs.shape)
Y = (imgs if imgs.ndim == 3 else imgs[..., 0]).astype(int)

conf = Config2D(
    n_rays=n_rays,
    n_channel_in=n_channel,
    use_gpu=False,
    train_epochs=4,
    train_steps_per_epoch=3,
    train_batch_size=2,
    # TODO adapt patch size to model
    train_patch_size=(128, 128),
)

# TODO clean this up
aug = augmend.Augmend()
aug.add([augmend.FlipRot90(axis=(0, 1)), augmend.FlipRot90(axis=(0, 1))])
aug.add([augmend.IntensityScaleShift(
    scale=(0.6, 2), shift=(-0.2, 0.2)), augmend.Identity()])
aug.add([augmend.AdditiveNoise(0.02), augmend.Identity()])


def augmenter(x, y):
    return aug([x, y])


data_kwargs = dict(
    n_rays=conf.n_rays,
    patch_size=conf.train_patch_size,
    grid=conf.grid,
    shape_completion=conf.train_shape_completion,
    b=conf.train_completion_crop,
    use_gpu=conf.use_gpu,
    foreground_prob=conf.train_foreground_only,
    n_classes=conf.n_classes,
    sample_ind_cache=conf.train_sample_cache,
)

data_train = StarDistData2D(
    X,
    Y,
    # classes=classes,
    batch_size=conf.train_batch_size,
    augmenter=augmenter,
    length=conf.train_epochs * conf.train_steps_per_epoch,
    **data_kwargs
)

# X = np.moveaxis(X, -1, 1)

# X = torch.as_tensor(X, dtype=torch.float)
# Y = torch.as_tensor(Y)

# data = torch.utils.data.TensorDataset(X, Y)

# dataloader = torch.utils.data.DataLoader(
    # data,
    # batch_size=2,
# )

model = StarDistTorch2d(conf, name='stardist', basedir=str(tmpdir))

for batch in data_train:
    x = torch.as_tensor(np.moveaxis(batch[0][0], -1, 1), dtype=torch.float)
    y_prob = torch.as_tensor(np.moveaxis(
        batch[1][0], -1, 1), dtype=torch.float)
    y_dist = torch.as_tensor(np.moveaxis(
        batch[1][1][..., :conf.n_rays], -1, 1), dtype=torch.float)

    prob, dist = model(x)
    break

    # TODO loss functions
    # TODO backprop
