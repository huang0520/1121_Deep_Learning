# %% Prepare the environment
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # disable warnings and info

import imageio as iio
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import trange

SAMPLE_COL = 8
SAMPLE_ROW = 8
SAMPLE_NUM = SAMPLE_COL * SAMPLE_ROW

IMG_HEIGHT = 64
IMG_WIDTH = 64
IMG_CHANNEL = 3
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL)

# Dataset parameters
BATCH_SIZE = 512
BUF = 65536

# Training parameters
EPOCH = 256
Z_DIM = 128
LEARNING_RATE = 5e-5
BETA_1 = 0.5
BETA_2 = 0.9
LAMBDA = 10

# Other parameters
IMG_DIR = "./data/img_align_celeba_png"
OUTPUT_DIR = "./output"
AUTOTUNE = tf.data.experimental.AUTOTUNE

# %%
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# %% Build dataset
def load_image(img_path: str):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=IMG_CHANNEL)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    return img


img_names: list = os.listdir(IMG_DIR)
img_paths: list = [os.path.join(IMG_DIR, name) for name in img_names]

dataset = tf.data.Dataset.from_tensor_slices(img_paths)
dataset = (
    dataset.map(load_image, num_parallel_calls=AUTOTUNE)
    .shuffle(BUF)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTOTUNE)
)


# %% Utility function
def utPuzzle(imgs, row, col, path=None):
    h, w, c = imgs[0].shape
    out = np.zeros((h * row, w * col, c), np.uint8)
    for n, img in enumerate(imgs):
        j, i = divmod(n, col)
        out[j * h : (j + 1) * h, i * w : (i + 1) * w, :] = img

    if path is not None:
        iio.imwrite(path, out.squeeze())
    return out


def utMakeGif(imgs, fname, duration):
    imgs = imgs.squeeze()
    n = float(len(imgs)) / duration
    clip = mpy.VideoClip(lambda t: imgs[int(n * t)], duration=duration)
    clip.write_gif(fname, fps=n)


# %% Define models
def GAN(img_shape, z_dim):
    # x-shape
    xh, xw, xc = img_shape
    # z-shape
    zh = xh // 4
    zw = xw // 4

    # return Generator and Discriminator
    return keras.Sequential(
        [  # Generator
            keras.layers.Dense(units=1024, input_shape=(z_dim,)),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dense(units=zh * zw * 256),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Reshape(target_shape=(zh, zw, 256)),
            keras.layers.Conv2DTranspose(
                filters=32, kernel_size=5, strides=2, padding="SAME"
            ),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2DTranspose(
                filters=xc,
                kernel_size=5,
                strides=2,
                padding="SAME",
                activation=keras.activations.sigmoid,
            ),
        ]
    ), keras.Sequential(
        [  # Discriminator
            keras.layers.Conv2D(
                filters=32,
                kernel_size=5,
                strides=(2, 2),
                padding="SAME",
                input_shape=img_shape,
            ),
            keras.layers.LeakyReLU(),
            keras.layers.Conv2D(
                filters=128, kernel_size=5, strides=(2, 2), padding="SAME"
            ),
            # keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.Flatten(),
            keras.layers.Dense(units=1024),
            # keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.Dense(units=1),
        ]
    )


# %% Train step
G, D = GAN(IMG_SHAPE, Z_DIM)
optimizer_g = keras.optimizers.Adam(LEARNING_RATE, BETA_1, BETA_2)
optimizer_d = keras.optimizers.Adam(LEARNING_RATE, BETA_1, BETA_2)


@tf.function
def g_train_step(img_real):
    z = tf.random.normal([BATCH_SIZE, Z_DIM])

    with tf.GradientTape() as tape:
        img_fake = G(z, training=True)

        d_real = D(img_real, training=True)
        d_fake = D(img_fake, training=True)

        loss_g = tf.reduce_mean(-d_fake)

        eta = tf.random.uniform([BATCH_SIZE, 1, 1, 1], 0.0, 1.0)
        interpolated = img_real * eta + img_fake * (1 - eta)
        d_interpolated = D(interpolated, training=True)
        d_gradients = tf.gradients(d_interpolated, interpolated)[0]
        l2_norm = tf.sqrt(tf.reduce_sum(tf.square(d_gradients), axis=[1, 2, 3]))
        gradient_penalty = tf.square(l2_norm - 1.0)

        loss_d = tf.reduce_mean(d_fake - d_real + LAMBDA * gradient_penalty)

    gradient_g = tape.gradient(loss_g, G.trainable_variables)
    optimizer_g.apply_gradients(zip(gradient_g, G.trainable_variables))

    return loss_g, loss_d


@tf.function
def d_train_step(img_real):
    z = tf.random.normal([BATCH_SIZE, Z_DIM])

    with tf.GradientTape() as tape:
        img_fake = G(z, training=True)

        d_real = D(img_real, training=True)
        d_fake = D(img_fake, training=True)

        loss_g = tf.reduce_mean(-d_fake)

        eta = tf.random.uniform([BATCH_SIZE, 1, 1, 1], 0.0, 1.0)
        interpolated = img_real * eta + img_fake * (1 - eta)
        d_interpolated = D(interpolated, training=True)

        d_grads = tf.gradients(d_interpolated, interpolated)[0]

        l2_norm = tf.sqrt(tf.reduce_sum(tf.square(d_grads), axis=[1, 2, 3]))
        grad_penalty = tf.square(l2_norm - 1.0)

        loss_d = tf.reduce_mean(d_fake - d_real + LAMBDA * grad_penalty)

    gradient_d = tape.gradient(loss_d, D.trainable_variables)
    optimizer_d.apply_gradients(zip(gradient_d, D.trainable_variables))

    return loss_g, loss_d


train_step = (
    d_train_step,
    d_train_step,
    d_train_step,
    d_train_step,
    d_train_step,
    g_train_step,
)

num_critic = len(train_step)

# %% Train
loss_g_record = [None] * EPOCH
loss_d_record = [None] * EPOCH
sample_record = [None] * EPOCH
sample_raw = tf.random.normal([SAMPLE_NUM, Z_DIM])

ckpt = tf.train.Checkpoint(G=G, D=D)
manager = tf.train.CheckpointManager(ckpt, "./checkpoints", max_to_keep=3)

critic = 0
pbar = trange(EPOCH, desc="WGAN-GP", unit="epoch")
for epoch in pbar:
    loss_g_total = 0.0
    loss_d_total = 0.0

    for real_imgs in dataset:
        loss_g, loss_d = train_step[critic](real_imgs)
        critic = critic + 1 if critic < num_critic - 1 else 0
        loss_g_total += loss_g.numpy()
        loss_d_total += loss_d.numpy()

    loss_g_record[epoch] = loss_g_total / len(dataset)
    loss_d_record[epoch] = loss_d_total / len(dataset)
    pbar.set_postfix(
        {
            "Loss G": "%.4f" % loss_g_record[epoch],
            "Loss D": "%.4f" % loss_d_record[epoch],
        }
    )

    out = G(sample_raw, training=False)
    img = utPuzzle(
        (out * 255.0).numpy().astype(np.uint8),
        SAMPLE_COL,
        SAMPLE_ROW,
        f"{OUTPUT_DIR}/gan_{epoch:04d}.png",
    )
    sample_record[epoch] = img
    manager.save()

    if (epoch + 1) % 32 == 0:
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Epoch {epoch}")
        plt.show()

# %%
plt.plot(range(EPOCH), loss_g_record, color="red", label="Generator Loss")
plt.plot(range(EPOCH), loss_d_record, color="blue", label="Discriminator Loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("WGAN-GP Training Loss")
plt.tight_layout()
plt.show()
