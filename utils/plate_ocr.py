import numpy as np
from dataclasses import dataclass

import onnxruntime as rt

CHAR_DICT = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "皖": 10,
    "京": 11,
    "渝": 12,
    "闽": 13,
    "甘": 14,
    "粤": 15,
    "桂": 16,
    "贵": 17,
    "琼": 18,
    "冀": 19,
    "黑": 20,
    "豫": 21,
    "鄂": 22,
    "湘": 23,
    "蒙": 24,
    "苏": 25,
    "赣": 26,
    "吉": 27,
    "辽": 28,
    "宁": 29,
    "青": 30,
    "陕": 31,
    "鲁": 32,
    "沪": 33,
    "晋": 34,
    "川": 35,
    "津": 36,
    "藏": 37,
    "新": 38,
    "云": 39,
    "浙": 40,
    "A": 41,
    "B": 42,
    "C": 43,
    "D": 44,
    "E": 45,
    "F": 46,
    "G": 47,
    "H": 48,
    "J": 49,
    "K": 50,
    "L": 51,
    "M": 52,
    "N": 53,
    "P": 54,
    "Q": 55,
    "R": 56,
    "S": 57,
    "T": 58,
    "U": 59,
    "V": 60,
    "W": 61,
    "X": 62,
    "Y": 63,
    "Z": 64,
    "": 65,
}


@dataclass
class OCRConfig:
    model_path: str = ""
    input_size: tuple = (128, 32)
    mean_values: tuple = (128.0, 128.0, 128.0)
    std_values: tuple = (128.0, 128.0, 128.0)

    # for onnx session
    log_severity_level: int = 3


class PlateOCR:
    def __init__(self, config: OCRConfig):
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

    def postprocess(self, outputs):
        pred = outputs[0]

        # Get the result
        pred = pred.squeeze()
        # softmax
        pred = np.exp(pred) / np.sum(np.exp(pred), axis=1, keepdims=True)
        out_char_codes = [np.argmax(pred[i]) for i in range(pred.shape[0])]

        out_str = ""
        no_character_code = len(CHAR_DICT) - 1

        for char_code in out_char_codes:
            out_str += list(CHAR_DICT.keys())[char_code]

        out_char_probs = pred[np.arange(pred.shape[0]), out_char_codes]
        out_char_probs = out_char_probs[out_char_codes != no_character_code]
        prob = np.prod(out_char_probs)

        return out_str, prob

    def preprocess(self, img):
        # preprocess the image
        img = img.astype(np.float32)
        img -= self.config.mean_values
        img /= self.config.std_values
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)

        return img
