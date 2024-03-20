# Annotations for AOLP Dataset

Due to a lack of keypoint annotations for license plates in the AOLP dataset, in this paper, we annotated keypoints for AOLP and simultaneously verified the accuracy of character annotations. Researchers can download the keypoint annotations for the "AC", "LE", and "RP" subsets from this repository. However, the original image data must be obtained by contacting the author via the email provided in the original repository.

Data Annotation Format:

```plain
<plate_number> <x1> <y1> <x2> <y2> <p1_x> <p1_y> <p2_x> <p2_y> <p3_x> <p3_y> <p4_x> <p4_y>
```

`<plate_number>`: Characters on the license plate.
`<x1> <y1> <x2> <y2>`: Original bounding box annotations in AOLP.
`<p1_x> … <p4_y>`: Coordinates of the four corners starting from the bottom right and moving clockwise.
