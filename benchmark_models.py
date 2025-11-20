from unet_model import UnetS4
from glob import glob
from pathlib import Path
from tifffile import imread
import time


# dataset
X = sorted(glob(r'c:\Users\kevin\Documents\Workspace\MantalysFrance\Data\dsb2018/test/images/*.tif'))
Y = sorted(glob(r'c:\Users\kevin\Documents\Workspace\MantalysFrance\Data\dsb2018/test/masks/*.tif'))
assert all(Path(x).name==Path(y).name for x,y in zip(X,Y))
X = list(map(imread,X))
Y = list(map(imread,Y))
n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]

models = {
    "MantaNet-24": "models/hybrid/stardist_r4_f24/model.onnx",
    "MantaNet-32": "models/hybrid/stardist_r4_f32/model.onnx",
}

times_per_image = {}

for model_name, model_path in models.items():
    print(f"Benchmarking model: {model_name}")
    model = UnetS4(model_path, intensity_range=(0, 1))

    start_time = time.time()
    for i in range(len(X)):
        model.forward(X[i])
    end_time = time.time()
    times_per_image[model_name] = (end_time - start_time) / len(X)
    print(f"Elapsed time per images: {times_per_image[model_name]} seconds")


# bar plot the time processing
import matplotlib.pyplot as plt
plt.bar(times_per_image.keys(), times_per_image.values())
plt.ylabel("Time per image (seconds)")
plt.show()
