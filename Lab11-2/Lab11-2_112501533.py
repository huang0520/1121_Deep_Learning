# %%
import functools
import os
import random
import time
from pathlib import Path

import IPython.display as display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import pyplot
from PIL import Image

mpl.rcParams["figure.figsize"] = (12, 12)
mpl.rcParams["axes.grid"] = False


# %% Define image loading and visualization functions
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    # in order to use CNN, add one additional dimension
    # to the original image
    # img shape: [height, width, channel] -> [batch_size, height, width, channel]
    img = img[tf.newaxis, :]

    return img


def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


# %% Load images
content_path = "./data/content.jpeg"
style_path = "./data/style.jpeg"

content_image = load_img(content_path)
style_image = load_img(style_path)

plt.subplot(1, 2, 1)
imshow(content_image, "Content Image")

plt.subplot(1, 2, 2)
imshow(style_image, "Style Image")

# %% Define content and style representation
# TODO: Change the content and style layers, and calculate the content, style loss
content_layers = ["block5_conv2"]

style_layers = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


# %% Build the model
def vgg_layers(layer_names):
    """Creates a VGG model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on ImageNet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image * 255)

# Look at the statistics of each layer's output
for name, output in zip(style_layers, style_outputs):
    print(name)
    print("  shape: ", output.numpy().shape)
    print("  min: ", output.numpy().min())
    print("  max: ", output.numpy().max())
    print("  mean: ", output.numpy().mean())
    print()


# %% Calculate style
def gram_matrix(input_tensor):
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)


# %% Extract style and content
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (
            outputs[: self.num_style_layers],
            outputs[self.num_style_layers :],
        )

        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        content_dict = {
            content_name: value
            for content_name, value in zip(self.content_layers, content_outputs)
        }

        style_dict = {
            style_name: value
            for style_name, value in zip(self.style_layers, style_outputs)
        }

        return {"content": content_dict, "style": style_dict}


# %%
extractor = StyleContentModel(style_layers, content_layers)

results = extractor(tf.constant(content_image))

print("Styles:")
for name, output in sorted(results["style"].items()):
    print("  ", name)
    print("    shape: ", output.numpy().shape)
    print("    min: ", output.numpy().min())
    print("    max: ", output.numpy().max())
    print("    mean: ", output.numpy().mean())
    print()

print("Contents:")
for name, output in sorted(results["content"].items()):
    print("  ", name)
    print("    shape: ", output.numpy().shape)
    print("    min: ", output.numpy().min())
    print("    max: ", output.numpy().max())
    print("    mean: ", output.numpy().mean())


# %% Run gradient descent
def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


style_targets = extractor(style_image)["style"]
content_targets = extractor(content_image)["content"]
style_weight = 1e-2
content_weight = 1e4
total_variation_weight = 30

image = tf.Variable(content_image)
opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)


def style_content_loss(outputs):
    style_outputs = outputs["style"]
    content_outputs = outputs["content"]
    style_loss = tf.add_n(
        [
            tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
            for name in style_outputs.keys()
        ]
    )
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n(
        [
            tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
            for name in content_outputs.keys()
        ]
    )
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss


# %% Define total variation loss
def high_pass_x_y(image):
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

    return x_var, y_var


def total_variation_loss(image):
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))


# %% Define train step
@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        loss += total_variation_weight * total_variation_loss(image)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))


# %% Train
opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
image = tf.Variable(content_image)

start = time.time()

epochs = 10
steps_per_epoch = 100

step = 0
for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1
        train_step(image)
    imshow(image.read_value())
    plt.title("Train step: {}".format(step))
    plt.show()

end = time.time()
print("Total time: {:.1f}".format(end - start))

# %% Save the result
file_name = "./output/style_transfer_nthu_starry_night.png"
mpl.image.imsave(file_name, image[0].numpy())


# * ====================================================================================
# %% AdalN

CONTENT_DIRS = ["./data/mscoco/test2014/"]
STYLE_DIRS = ["./data/wikiart/test/"]

IMG_MEANS = np.array([103.939, 116.779, 123.68])  # BGR

IMG_SHAPE = (224, 224, 3)  # training image shape, (h, w, c)
SHUFFLE_BUFFER = 1000
BATCH_SIZE = 32
EPOCHS = 30
STEPS_PER_EPOCH = 12000 // BATCH_SIZE


# %% Functions for sampling and plotting images
def sample_files(dir, num, pattern="**/*.jpg"):
    """Samples files in a directory using the reservoir sampling."""

    paths = Path(dir).glob(pattern)  # list of Path objects
    sampled = []
    for i, path in enumerate(paths):
        if i < num:
            sampled.append(path)
        else:
            s = random.randint(0, i)
            if s < num:
                sampled[s] = path
    return sampled


def plot_images(dir, row, col, pattern):
    paths = sample_files(dir, row * col, pattern)

    plt.figure(figsize=(2 * col, 2 * row))
    for i in range(row * col):
        im = Image.open(paths[i])
        w, h = im.size

        plt.subplot(row, col, i + 1)
        plt.imshow(im)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(f"{w}x{h}")
    plt.show()


print("Sampled content images:")
plot_images(CONTENT_DIRS[0], 4, 8, pattern="*.jpg")

print("Sampled style images:")
plot_images(STYLE_DIRS[0], 4, 8, pattern="*.jpg")


# %% Clean up the dataset
"""
def clean(dir_path, min_shape=None):
    paths = Path(dir_path).glob("**/*.jpg")
    deleted = 0
    for path in paths:
        try:
            # Make sure we can decode the image
            im = tf.io.read_file(str(path.resolve()))
            im = tf.image.decode_jpeg(im)

            # Remove grayscale images
            shape = im.shape
            if shape[2] < 3:
                path.unlink()
                deleted += 1

            # Remove small images
            if min_shape is not None:
                if shape[0] < min_shape[0] or shape[1] < min_shape[1]:
                    path.unlink()
                    deleted += 1
        except Exception as e:
            path.unlink()
            deleted += 1
    return deleted


for dir in CONTENT_DIRS:
    deleted = clean(dir)
print(f"#Deleted content images: {deleted}")

for dir in STYLE_DIRS:
    deleted = clean(dir)
print(f"#Deleted style images: {deleted}")
"""


# %% Prepare and build the dataset
def preprocess_image(path, init_shape=(448, 448)):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, init_shape)
    image = tf.image.random_crop(image, size=IMG_SHAPE)
    image = tf.cast(image, tf.float32)

    # Convert image from RGB to BGR, then zero-center each color channel with
    # respect to the ImageNet dataset, without scaling.
    image = image[..., ::-1]  # RGB to BGR
    image -= (103.939, 116.779, 123.68)  # BGR means
    return image


def np_image(image):
    image += (103.939, 116.779, 123.68)  # BGR means
    image = image[..., ::-1]  # BGR to RGB
    image = tf.clip_by_value(image, 0, 255)
    image = tf.cast(image, dtype="uint8")
    return image.numpy()


def build_dataset(num_gpus=1):
    c_paths = []
    for c_dir in CONTENT_DIRS:
        c_paths += Path(c_dir).glob("*.jpg")
    c_paths = [str(path.resolve()) for path in c_paths]
    s_paths = []
    for s_dir in STYLE_DIRS:
        s_paths += Path(s_dir).glob("*.jpg")
    s_paths = [str(path.resolve()) for path in s_paths]
    print(
        f"Building dataset from {len(c_paths):,} content images and {len(s_paths):,} style images... ",
        end="",
    )

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    c_ds = tf.data.Dataset.from_tensor_slices(c_paths)
    c_ds = c_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    c_ds = c_ds.repeat()
    c_ds = c_ds.shuffle(buffer_size=SHUFFLE_BUFFER)

    s_ds = tf.data.Dataset.from_tensor_slices(s_paths)
    s_ds = s_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    s_ds = s_ds.repeat()
    s_ds = s_ds.shuffle(buffer_size=SHUFFLE_BUFFER)

    ds = tf.data.Dataset.zip((c_ds, s_ds))
    ds = ds.batch(BATCH_SIZE * num_gpus)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    print("done")
    return ds


# %%
ds = build_dataset()
c_batch, s_batch = next(iter(ds.take(1)))

print("Content batch shape:", c_batch.shape)
print("Style batch shape:", s_batch.shape)

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(np_image(c_batch[0]))
plt.grid(False)
plt.xticks([])
plt.yticks([])
plt.xlabel("Content")

plt.subplot(1, 2, 2)
plt.imshow(np_image(s_batch[0]))
plt.grid(False)
plt.xticks([])
plt.yticks([])
plt.xlabel("Style")

plt.show()


# %% Define AdaIN layer
class AdaIN(tf.keras.layers.Layer):
    def __init__(self, name, epsilon=1e-5):
        super(AdaIN, self).__init__()
        self.epsilon = epsilon

    def call(self, inputs):
        content = inputs[0]
        style = inputs[1]

        # Calculate the mean and standard deviation of the content
        c_mean, c_var = tf.nn.moments(content, axes=(1, 2), keepdims=True)
        c_std = tf.sqrt(c_var + self.epsilon)

        # Calculate the mean and standard deviation of the style
        s_mean, s_var = tf.nn.moments(style, axes=(1, 2), keepdims=True)
        s_std = tf.sqrt(s_var + self.epsilon)

        # Normalize the content (Remove style)
        c_norm = (content - c_mean) / c_std

        # Apply the style
        return s_std * c_norm + s_mean


# %% Define the model
class ArbitraryStyleTransferNet(tf.keras.Model):
    CONTENT_LAYER = "block4_conv1"
    STYLE_LAYERS = ("block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1")

    @staticmethod
    def declare_decoder():
        a_input = tf.keras.Input(shape=(28, 28, 512), name="input_adain")

        h = tf.keras.layers.Conv2DTranspose(256, 3, padding="same", activation="relu")(
            a_input
        )
        h = tf.keras.layers.UpSampling2D(2)(h)
        h = tf.keras.layers.Conv2DTranspose(256, 3, padding="same", activation="relu")(
            h
        )
        h = tf.keras.layers.Conv2DTranspose(256, 3, padding="same", activation="relu")(
            h
        )
        h = tf.keras.layers.Conv2DTranspose(256, 3, padding="same", activation="relu")(
            h
        )
        h = tf.keras.layers.Conv2DTranspose(128, 3, padding="same", activation="relu")(
            h
        )
        h = tf.keras.layers.UpSampling2D(2)(h)
        h = tf.keras.layers.Conv2DTranspose(128, 3, padding="same", activation="relu")(
            h
        )
        h = tf.keras.layers.Conv2DTranspose(64, 3, padding="same", activation="relu")(h)
        h = tf.keras.layers.UpSampling2D(2)(h)
        h = tf.keras.layers.Conv2DTranspose(64, 3, padding="same", activation="relu")(h)
        output = tf.keras.layers.Conv2DTranspose(3, 3, padding="same")(h)

        return tf.keras.Model(inputs=a_input, outputs=output, name="decoder")

    def __init__(
        self,
        img_shape=(224, 224, 3),
        content_loss_weight=1,
        style_loss_weight=10,
        name="arbitrary_style_transfer_net",
        **kwargs,
    ):
        super(ArbitraryStyleTransferNet, self).__init__(name=name, **kwargs)

        self.img_shape = img_shape
        self.content_loss_weight = content_loss_weight
        self.style_loss_weight = style_loss_weight

        vgg19 = tf.keras.applications.VGG19(
            include_top=False, weights="imagenet", input_shape=img_shape
        )
        vgg19.trainable = False

        c_output = [vgg19.get_layer(ArbitraryStyleTransferNet.CONTENT_LAYER).output]
        s_outputs = [
            vgg19.get_layer(name).output
            for name in ArbitraryStyleTransferNet.STYLE_LAYERS
        ]
        self.vgg19 = tf.keras.Model(
            inputs=vgg19.input, outputs=c_output + s_outputs, name="vgg19"
        )
        self.vgg19.trainable = False

        self.adain = AdaIN(name="adain")
        self.decoder = ArbitraryStyleTransferNet.declare_decoder()

    def call(self, inputs):
        c_batch, s_batch = inputs

        c_enc = self.vgg19(c_batch)
        c_enc_c = c_enc[0]

        s_enc = self.vgg19(s_batch)
        s_enc_c = s_enc[0]
        s_enc_s = s_enc[1:]

        # normalized_c is the output of AdaIN layer
        normalized_c = self.adain((c_enc_c, s_enc_c))
        output = self.decoder(normalized_c)

        # Calculate loss
        out_enc = self.vgg19(output)
        out_enc_c = out_enc[0]
        out_enc_s = out_enc[1:]

        loss_c = tf.reduce_mean(tf.math.squared_difference(out_enc_c, normalized_c))
        self.add_loss(self.content_loss_weight * loss_c)

        loss_s = 0
        for o, s in zip(out_enc_s, s_enc_s):
            o_mean, o_var = tf.nn.moments(o, axes=(1, 2), keepdims=True)
            o_std = tf.sqrt(o_var + self.adain.epsilon)

            s_mean, s_var = tf.nn.moments(s, axes=(1, 2), keepdims=True)
            s_std = tf.sqrt(s_var + self.adain.epsilon)

            loss_mean = tf.reduce_mean(tf.math.squared_difference(o_mean, s_mean))
            loss_std = tf.reduce_mean(tf.math.squared_difference(o_std, s_std))

            loss_s += loss_mean + loss_std
        self.add_loss(self.style_loss_weight * loss_s)

        return output, c_enc_c, normalized_c, out_enc_c


# %% Function for plotting images
# Plot results
def plot_outputs(outputs, captions=None, col=5):
    row = len(outputs)
    plt.figure(figsize=(3 * col, 3 * row))
    for i in range(col):
        for j in range(row):
            plt.subplot(row, col, j * col + i + 1)
            plt.imshow(np_image(outputs[j][i, ..., :3]))
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            if captions is not None:
                plt.xlabel(captions[j])
    plt.show()


# %% Build the model
ds = build_dataset()
model = ArbitraryStyleTransferNet(img_shape=IMG_SHAPE)

c_batch, s_batch = next(iter(ds.take(1)))
print(f"Input shape: ({c_batch.shape}, {s_batch.shape})")
output, *_ = model((c_batch, s_batch))
print(f"Output shape: {output.shape}")
print(f"Init. content loss: {model.losses[0]:,.2f}, style loss: {model.losses[1]:,.2f}")
model.summary()

# %% Training
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
c_loss_metric, s_loss_metric = tf.keras.metrics.Mean(), tf.keras.metrics.Mean()

CKP_DIR = "checkpoints"
init_epoch = 1

ckp = tf.train.latest_checkpoint(CKP_DIR)
if ckp:
    model.load_weights(ckp)
    init_epoch = int(ckp.split("_")[-1]) + 1
    print(f"Resume training from epoch {init_epoch-1}")


# %% Train step
@tf.function
def train_step(inputs):
    with tf.GradientTape() as tape:
        model(inputs)
        c_loss, s_loss = model.losses
        loss = c_loss + s_loss
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    c_loss_metric(c_loss)
    s_loss_metric(s_loss)


# %% Training loop
def train(dataset, init_epoch):
    for epoch in range(init_epoch, EPOCHS + 1):
        print(f"Epoch {epoch:>2}/{EPOCHS}")
        for step, inputs in enumerate(dataset.take(STEPS_PER_EPOCH)):
            train_step(inputs)
            print(
                f"{step+1:>5}/{STEPS_PER_EPOCH} - loss: {c_loss_metric.result()+s_loss_metric.result():,.2f} - content loss: {c_loss_metric.result():,.2f} - style loss: {s_loss_metric.result():,.2f}",
                end="\r",
            )

        print()
        model.save_weights(os.path.join(CKP_DIR, f"ckpt_{epoch}"))
        c_loss_metric.reset_states()
        s_loss_metric.reset_states()

        output, c_enc_c, normalized_c, out_enc_c = model((c_batch, s_batch))
        plot_outputs(
            (s_batch, c_batch, output, c_enc_c, normalized_c, out_enc_c),
            ("Style", "Content", "Trans", "Content Enc", "Normalized", "Trans Enc"),
        )


# %% Train
train(ds, init_epoch)
