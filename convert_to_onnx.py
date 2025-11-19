"""
A utils script to convert a Stardist model (h5) to ONNX format.
"""

import tensorflow as tf
from stardist.models import StarDist2D
import onnx
from tf2onnx.convert import from_keras




n_rays = 4 # Number of rays in the Stardist model, default is 32
n_filters = 12 # Number of filters in the first convolutional layer, default is 32
layer_type = "sepconv"  # Type of layers used in the model, either "unet" or "resnet"

basedir = "models\hybrid"
model_name = f"unet-{layer_type}-{n_rays}-{n_filters}"

print(f"Converting Stardist model {model_name} to ONNX format...")
model = StarDist2D(config=None, name=model_name, basedir=basedir)
print(model.keras_model.summary())

input_signature = [tf.TensorSpec([None, None, None, 1], tf.float32, name='input')]
print(f"Input signature: {input_signature}")

onnx_target = f"{basedir}/{model_name}/model.onnx"
print(f"Saving ONNX model to {onnx_target}...")
onnx_model, _ = from_keras(model.keras_model, input_signature, opset=17)
onnx.save_model(onnx_model, onnx_target)
print(f"ONNX model saved successfully to {onnx_target}.")


