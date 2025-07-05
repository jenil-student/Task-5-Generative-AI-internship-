import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
import os

# --- Helper Functions ---
def load_img(path_to_img, max_dim=512):
    img = Image.open(path_to_img)
    long = max(img.size)
    scale = max_dim / long
    # Use Image.Resampling.LANCZOS for compatibility with Pillow >= 10
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), resample)
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img, dtype=tf.float32)

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return Image.fromarray(tensor)

# --- Style Transfer Core ---
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super().__init__()
        self.vgg = self.vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def vgg_layers(self, layer_names):
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in layer_names]
        model = tf.keras.Model([vgg.input], outputs)
        return model

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])
        style_outputs = [self.gram_matrix(style_output) for style_output in style_outputs]
        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}

    def gram_matrix(self, input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return result / num_locations

# --- Streamlit App ---
st.set_page_config(page_title="Neural Style Transfer", layout="centered")
st.title("Neural Style Transfer")
st.write("Apply the artistic style of one image to the content of another using deep learning.")

content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"], key="content")
style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"], key="style")

if content_file and style_file:
    content_image = load_img(content_file)
    style_image = load_img(style_file)

    content_layers = ['block5_conv2']
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    extractor = StyleContentModel(style_layers, content_layers)
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    image = tf.Variable(content_image)
    opt = tf.optimizers.Adam(learning_rate=0.02)
    style_weight = 1e-2
    content_weight = 1e4

    @tf.function
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            style_loss = tf.add_n([tf.reduce_mean((outputs['style'][name]-style_targets[name])**2)
                                   for name in outputs['style'].keys()])
            style_loss *= style_weight / len(style_layers)
            content_loss = tf.add_n([tf.reduce_mean((outputs['content'][name]-content_targets[name])**2)
                                     for name in outputs['content'].keys()])
            content_loss *= content_weight / len(content_layers)
            loss = style_loss + content_loss
        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(tf.clip_by_value(image, 0.0, 1.0))

    st.write("Processing... (this may take up to a minute)")
    for n in range(50):
        train_step(image)
    result_image = tensor_to_image(image.numpy())
    st.subheader("Content Image")
    st.image(tensor_to_image(content_image), use_container_width=True)
    st.subheader("Style Image")
    st.image(tensor_to_image(style_image), use_container_width=True)
    st.subheader("Stylized Output")
    st.image(result_image, use_container_width=True)
else:
    st.info("Please upload both a content and a style image.")
