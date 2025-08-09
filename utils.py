import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# === Custom Layers and Model ===
class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(name="scale", shape=(input_shape[-1],), initializer="ones", trainable=True)
        self.offset = self.add_weight(name="offset", shape=(input_shape[-1],), initializer="zeros", trainable=True)

    def call(self, x):
        mean, var = tf.nn.moments(x, [1, 2], keepdims=True)
        return self.scale * (x - mean) / tf.sqrt(var + self.epsilon) + self.offset

class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, activation=True):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding='same')
        self.norm = InstanceNormalization()
        self.activation = tf.keras.layers.ReLU() if activation else tf.identity

    def call(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.activation(x)

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = ConvLayer(filters, 3, 1)
        self.conv2 = ConvLayer(filters, 3, 1, activation=False)

    def call(self, x):
        return x + self.conv2(self.conv1(x))

class StyleTransferNet(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = tf.keras.Sequential([
            ConvLayer(32, 9, 1),
            ConvLayer(64, 3, 2),
            ConvLayer(128, 3, 2),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            tf.keras.layers.UpSampling2D(),
            ConvLayer(64, 3, 1),
            tf.keras.layers.UpSampling2D(),
            ConvLayer(32, 3, 1),
            tf.keras.layers.Conv2D(3, 9, 1, padding='same', activation='sigmoid')
        ])

    def call(self, inputs):
        return self.model(inputs)

# === Model Loaders ===
def load_custom_model(weights_path):
    model = StyleTransferNet()
    dummy_input = tf.zeros([1, 256, 256, 3])
    _ = model(dummy_input)
    model.load_weights(weights_path)
    return model

def load_saved_model(h5_weights_path):
    model = StyleTransferNet()
    dummy_input = tf.zeros([1, 256, 256, 3])
    _ = model(dummy_input)
    model.load_weights(h5_weights_path)
    return model


# === Image Enhancement & Filtering ===
def enhance_image(img_path, brightness=30, contrast=1.1, sat_boost_blue_yellow=1.3, sat_drop_others=0.9, white_boost=0.1):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image at path '{img_path}' could not be loaded.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    sat_scale = np.where(
        ((h >= 30) & (h <= 60)) | ((h >= 90) & (h <= 150)),
        sat_boost_blue_yellow,
        sat_drop_others
    )
    s = np.clip(s * sat_scale, 0, 255)
    hsv = cv2.merge([h.astype(np.uint8), s.astype(np.uint8), v.astype(np.uint8)])
    img_sat = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    img_bright = np.clip(img_sat + (brightness / 255.0), 0, 1)
    img_contrast = np.clip(0.5 + contrast * (img_bright - 0.5), 0, 1)
    img_white = img_contrast + white_boost * (1.0 - img_contrast)
    img_white = np.clip(img_white, 0, 1)

    return (img * 255).astype(np.uint8), (img_white * 255).astype(np.uint8)

def grayscale_except_blue_yellow(img,
                                  blue_range=((80, 30, 30), (140, 255, 255)),
                                  yellow_range=((10, 30, 30), (50, 255, 255))):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask_blue = cv2.inRange(hsv, blue_range[0], blue_range[1])
    mask_yellow = cv2.inRange(hsv, yellow_range[0], yellow_range[1])
    mask = cv2.bitwise_or(mask_blue, mask_yellow)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    result = np.where(mask[..., None] > 0, img, gray_rgb)
    return result

# === Image Preprocessing ===
def load_and_preprocess(input, is_array=False):
    if not is_array:
        img = Image.open(input).convert('RGB')
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize((256, 256))
        img = np.array(img).astype(np.float32) / 255.0
    else:
        h, w, _ = input.shape
        side = min(h, w)
        top = (h - side) // 2
        left = (w - side) // 2
        img = input[top:top+side, left:left+side]
        img = cv2.resize(img, (256, 256)).astype(np.float32) / 255.0
    return tf.convert_to_tensor(img)[tf.newaxis, ...]
