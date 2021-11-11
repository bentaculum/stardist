import numpy as np
from stardist.models import Config2D, StarDist2D
from csbdeep.utils.tf import keras_import
Sequence = keras_import('utils', 'Sequence')


class NumpySequence(Sequence):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, n):
        return self.data[n]

    def __len__(self):
        return len(self.data)


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


tmpdir = "./tmp"
n_rays = 17
grid = (1, 1)
n_channel = 1
workers = 1
use_sequence = True

img = circle_image(shape=(160, 160))
imgs = np.repeat(img[np.newaxis], 3, axis=0)

if n_channel is not None:
    imgs = np.repeat(imgs[..., np.newaxis], n_channel, axis=-1)
else:
    n_channel = 1

X = imgs + .6 * np.random.uniform(0, 1, imgs.shape)
Y = (imgs if imgs.ndim == 3 else imgs[..., 0]).astype(int)


if use_sequence:
    X, Y = NumpySequence(X), NumpySequence(Y)

conf = Config2D(
    n_rays=n_rays,
    grid=grid,
    n_channel_in=n_channel,
    use_gpu=False,
    train_epochs=2,
    train_steps_per_epoch=1,
    train_batch_size=2,
    train_loss_weights=(4, 1),
    train_patch_size=(128, 128),
    train_sample_cache=not use_sequence
)

model = StarDist2D(conf, name='stardist', basedir=str(tmpdir))
model.train(X, Y, validation_data=(X[:2], Y[:2]), workers=workers)
ref = model.predict(X[0])
res = model.predict(X[0], n_tiles=(
    (2, 3) if X[0].ndim == 2 else (2, 3, 1)))
