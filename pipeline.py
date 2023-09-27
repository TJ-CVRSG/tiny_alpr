from pathlib import Path

import cv2
import numpy as np

from utils.utils import get_max_box, get_square_coordinates, adjust_boundaries
from utils.plate_det import PlateDet, DetConfig
from utils.plate_loc import PlateLoc, PlateLocConfig
from utils.plate_ocr import PlateOCR, OCRConfig

det_config = DetConfig(
    model_path="./models/ccpd/yolov5t_ghost_320.onnx",
    input_size=(320, 320),
    conf_thres=0.1,
)
plate_det = PlateDet(det_config)

loc_config = PlateLocConfig(
    model_path="./models/ccpd/mobilenetv2_025_rle_128.onnx",
    input_size=(128, 128),
)
plate_loc = PlateLoc(loc_config)

ocr_config = OCRConfig(
    model_path="./models/ccpd/lprnet_cls_mbv2_05.onnx",
    input_size=(128, 32),
)
plate_ocr = PlateOCR(ocr_config)

image_dir = "./images"

if __name__ == "__main__":
    image_paths = sorted(list(Path(image_dir).glob("*.jpg")))

    for image_path in image_paths:
        origin_img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)

        # PLATE DETECTION
        det_input = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
        det_input = cv2.resize(det_input, det_config.input_size)

        boxes, scores, classes = plate_det.detect(det_input)

        if len(boxes) == 0:
            print("No plate detected")
            continue

        # rescale the boxes to the origin image
        boxes /= np.array(det_config.input_size[::-1] * 2)
        boxes *= np.array(origin_img.shape[1::-1] * 2)
        boxes = boxes.astype(np.int32)

        max_box, max_score, max_class = get_max_box(boxes, scores, classes)
        x1, y1, x2, y2 = get_square_coordinates(max_box, origin_img.shape)
        (
            x1,
            y1,
            x2,
            y2,
            top_pad,
            bottom_pad,
            left_pad,
            right_pad,
        ) = adjust_boundaries(x1, y1, x2, y2, origin_img.shape)

        # crop the square area
        crop_img = origin_img[y1:y2, x1:x2]
        crop_img = cv2.copyMakeBorder(
            crop_img,
            top_pad,
            bottom_pad,
            left_pad,
            right_pad,
            cv2.BORDER_CONSTANT,
        )

        # PLATE LOCALIZATION
        reg_input = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        reg_input = cv2.resize(reg_input, loc_config.input_size)

        keypoints, _ = plate_loc.predict(reg_input)

        # PLATE RECOGNITION PART
        # perform warp perspective
        src_pts = keypoints * np.array([crop_img.shape[1], crop_img.shape[0]])
        src_pts = src_pts.astype(np.float32)
        dst_pts = np.array(
            [[128, 32], [0, 32], [0, 0], [128, 0]],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warp_img = cv2.warpPerspective(crop_img, M, ocr_config.input_size)

        ocr_input = cv2.cvtColor(warp_img, cv2.COLOR_BGR2RGB)
        plate_number, prob = plate_ocr.predict(ocr_input)

        print(f"Plate number: {plate_number}, Prob: {prob}")
        cv2.imshow("origin", origin_img)
        cv2.imshow("crop", crop_img)
        cv2.imshow("warp", warp_img)
        cv2.waitKey(0)
