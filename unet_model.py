import numpy as np
import cv2
import time
import onnxruntime as rt
from skimage.morphology import disk
from skimage.segmentation import watershed





DISK_1 = disk(1)
DISK_2 = disk(2)

SOBEL_4 = np.array(
    [
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
    ],
    dtype=np.float32,
)


def normalize(x: np.ndarray, in_range: list = None, dtype=np.float32):
    if in_range is not None:
        imin = in_range[0] if in_range[0] is not None else np.min(x)
        imax = in_range[1] if in_range[1] is not None else np.max(x)
        x = np.clip(x, imin, imax)
    else:
        imin = np.min(x)
        imax = np.max(x)
    x = (x - imin) / (imax - imin)
    if dtype == np.uint8:
        x = (x * 255).astype(dtype)
    return x


def normalize_channels(x: np.ndarray, in_ranges: list, dtype=np.float32):
    channel_list = []
    for i in range(x.shape[0]):
        channel = normalize(x[i], in_ranges[i], dtype=dtype)
        channel_list.append(channel)
    x = np.stack(channel_list, axis=0)
    return x


def resize_multiple_of(x, x_height, x_width, multiple):
    resized = False
    target_shape = ((x_height // multiple) * multiple, (x_width // multiple) * multiple)
    if x_height != target_shape[0] or x_width != target_shape[1]:
        # Resize the image to the target shape
        x = cv2.resize(x, dsize=target_shape, interpolation=cv2.INTER_NEAREST)
        resized = True
    return x, resized


def to_tensor(x):
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    x = np.expand_dims(x, axis=-1)  # Add channel dimension
    return x


def filter_by_size(label_image, size_filter):
    if size_filter <= 0:
        return label_image

    counts = np.bincount(label_image.ravel())
    small = np.where((counts < size_filter))[0]
    if len(small) > 0:
        label_image[np.isin(label_image, small)] = 0
    return label_image


def filter_by_size_holes(mask_image, size_filter):
    if size_filter <= 0:
        return mask_image

    holes_mask = (mask_image == 0).astype(np.uint8)
    holes = cv2.connectedComponents(holes_mask, connectivity=8)[1]
    retained_holes = filter_by_size(holes, size_filter)
    mask_image[holes_mask & (retained_holes == 0)] = 1
    return mask_image


def postprocess_unet_features(features, size_filter=16):
    [probability, magnitude] = features

    # apply gaussian smoothing
    probability = cv2.GaussianBlur(probability, sigmaX=1, sigmaY=1, ksize=(5, 5))

    probability = normalize(probability, in_range=(0.1, None))

    markers_mask = ((probability > 0) & (magnitude < 0.1)).astype(np.uint8)
    markers_mask = cv2.morphologyEx(markers_mask, cv2.MORPH_OPEN, DISK_1)
    markers = cv2.connectedComponents(markers_mask, connectivity=4)[1]

    nuclei = watershed(
        magnitude,
        markers=markers,
        mask=probability > 0,
        compactness=1,
        # watershed_line=True,
        connectivity=2,
    )

    time_start = time.time()
    nuclei = filter_by_size(nuclei, size_filter)
    time_end = time.time()
    print(f"Elapsed speck removal: {time_end - time_start} seconds")
    return nuclei


def predict_unet_features(batch):
    # TODO: optimize this function later; include into the ONNX model if possible
    # keep probabilities as is, but compute the magnitude of the gradients
    batch_probability = batch[0]  # Assuming batch[0] is the probability map
    # batch_probability[batch_probability < 0.005] = 0  # Ensure no negative values
    batch_gradients = batch[1]  # Assuming batch[1] contains the 32 gradients

    batch_features = np.zeros(
        (
            2,
            len(batch_probability),
            batch_probability.shape[1],
            batch_probability.shape[2],
        ),
        dtype=np.float32,
    )
    batch_features[0] = batch_probability[
        :, :, :, 0
    ]  # Probability map, assuming the first channel is the probability
    for tile_index in range(len(batch_probability)):
        filtered = [
            cv2.filter2D(batch_gradients[tile_index, :, :, i], cv2.CV_32F, SOBEL_4[i])
            for i in range(4)
        ]
        magnitude = np.max(filtered, axis=0)
        magnitude = normalize(magnitude, in_range=(0, None))
        batch_features[1][tile_index] = magnitude  # Magnitude of gradients

    return batch_features


class OnnxModel:
    """
    Only the ONNX model loading and inference logic.
    """

    def __init__(self, model_path):
        self.session = rt.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.outputs = [output.name for output in self.session.get_outputs()]

    def forward(self, x):
        # x shape: (batch_size, height, width, channels)
        return self.session.run(output_names=self.outputs, input_feed={self.input_name: x})


class UnetS4(OnnxModel):
    def __init__(self, model_path, intensity_range):
        super().__init__(model_path)
        self.intensity_range = intensity_range

    def forward(self, x):
        # x shape: (height, width)

        # normalize input
        x = normalize(x, self.intensity_range).astype(np.float32)

        # Ensure the image is in the correct format for the model, multiple of 16
        # imposed by the UNet architecture
        x_height, x_width = x.shape[:2]
        x, resized = resize_multiple_of(x, x_height, x_width, multiple=16)

        x = to_tensor(x)  # shape (1, height, width, 1), ready for ONNX model

        unet_features = super().forward(x)  # TODO: investigate later to batch more tiles
        features = predict_unet_features(
            unet_features
        )  # get postprocessed features: probability and magnitude maps
        return features