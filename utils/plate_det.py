import cv2
import numpy as np
from dataclasses import dataclass

import onnxruntime as rt


@dataclass
class DetConfig:
    model_path: str = ""
    num_classes: int = 2
    input_size: tuple = (320, 320)
    conf_thres: float = 0.45
    iou_thres: float = 0.45
    classes: tuple = ("blue_plate", "green_plate")
    anchors: tuple = (
        ((10, 13), (16, 30), (33, 23)),
        ((30, 61), (62, 45), (59, 119)),
        ((116, 90), (156, 198), (373, 326)),
    )

    # for onnx session
    log_severity_level: int = 3


class PlateDet:
    def __init__(self, config: DetConfig):
        self.config = config

        # onnxruntime session
        so = rt.SessionOptions()
        so.log_severity_level = config.log_severity_level
        self.sess = rt.InferenceSession(
            config.model_path,
            so,
            providers=["CPUExecutionProvider"],
        )

        # input and output names
        self.input_name = self.sess.get_inputs()[0].name  # images
        self.output_names = [output.name for output in self.sess.get_outputs()]

    def detect(self, img):
        # preprocess the image
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        # inference
        outputs = self.sess.run(self.output_names, {self.input_name: img})

        # postprocess the outputs
        boxes, scores, classes = self.postprocess(outputs)

        return boxes, scores, classes

    def postprocess(self, outputs):
        boxes, scores, classes = [], [], []

        # reshape the outputs to # (anchors, classes+5, h, w)
        # assume the batch size is 1
        outputs = [
            out.reshape(
                [len(self.config.anchors[i]), -1] + list(out.shape[-2:])
            )  # noqa
            for i, out in enumerate(outputs)
        ]

        # for each output
        for i, out in enumerate(outputs):
            box_pred = out[:, :4, :, :]
            score_pred = out[:, 4:5, :, :]
            class_pred = out[:, 5:, :, :]

            # get the box
            box = self.boxpred2box(box_pred, np.array(self.config.anchors[i]))
            box = box.transpose(0, 2, 3, 1)
            boxes.append(box.reshape(-1, 4))

            # get the score
            score = score_pred.transpose(0, 2, 3, 1)
            scores.append(
                score.reshape(
                    -1,
                )
            )

            # get the class
            class_pred = class_pred.transpose(0, 2, 3, 1)
            class_pred = class_pred.reshape(-1, self.config.num_classes)
            classes.append(class_pred)

        # concatenate the outputs
        boxes = np.concatenate(boxes)
        scores = np.concatenate(scores)
        classes = np.concatenate(classes)

        # filter the boxes
        mask = scores > self.config.conf_thres
        boxes = boxes[mask]
        scores = scores[mask]
        classes = classes[mask]

        # nms
        keep = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.config.conf_thres,
            self.config.iou_thres,
        )

        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep]

        # compute confidence and class id
        scores = classes.max(axis=1) * scores
        classes = classes.argmax(axis=1)

        return boxes, scores, classes

    def boxpred2box(self, box_pred, anchor):
        # build the grid and stride
        grid_h, grid_w = box_pred.shape[-2:]
        col, row = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
        col = col.reshape(1, 1, grid_h, grid_w).repeat(len(anchor), axis=0)
        row = row.reshape(1, 1, grid_h, grid_w).repeat(len(anchor), axis=0)
        grid = np.concatenate((col, row), axis=1).astype(np.float32)
        stride = np.array(
            [
                self.config.input_size[0] / grid_h,
                self.config.input_size[1] / grid_w,
            ]
        ).reshape(1, 2, 1, 1)
        anchor = anchor.reshape(*anchor.shape, 1, 1)

        # box_pred: (x, y, w, h)
        box_pred[:, :2, :, :] = box_pred[:, :2, :, :] * 2 - 0.5
        box_pred[:, :2, :, :] = (box_pred[:, :2, :, :] + grid) * stride
        box_pred[:, 2:4, :, :] = pow(box_pred[:, 2:4, :, :] * 2, 2) * anchor

        # box: (x1, y1, x2, y2)
        box = np.concatenate(
            (
                box_pred[:, :2, :, :] - box_pred[:, 2:4, :, :] / 2,
                box_pred[:, :2, :, :] + box_pred[:, 2:4, :, :] / 2,
            ),
            axis=1,
        )

        return box
