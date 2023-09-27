import numpy as np
from dataclasses import dataclass

import onnxruntime as rt


@dataclass
class PlateLocConfig:
    model_path: str = ""

    input_size: tuple = (160, 160)
    mean: tuple = (123.675, 116.28, 103.53)
    std: tuple = (58.395, 57.12, 57.375)

    # for onnx session
    log_severity_level: int = 3


class PlateLoc:
    def __init__(self, config: PlateLocConfig):
        self.config = config

        # onnxruntime session
        so = rt.SessionOptions()
        so.log_severity_level = config.log_severity_level
        self.sess = rt.InferenceSession(
            config.model_path,
            so,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [output.name for output in self.sess.get_outputs()]

    def predict(self, img):
        img = self.preprocess(img)
        outputs = self.sess.run(self.output_names, {self.input_name: img})
        return self.postprocess(outputs)

    def preprocess(self, img):
        img = img.astype(np.float32)
        img -= self.config.mean
        img /= self.config.std
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)
        return img

    def postprocess(self, outputs):
        keypoints, scores = outputs

        # perform sigmoid on scores
        scores = 1 / (1 + np.exp(-scores))
        scores = (1 - scores).mean(axis=-1)

        return keypoints, scores
