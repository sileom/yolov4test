# YoloV4 Test
The repository makes a YOLOV4 hole detection with grayscale images from a Intel Realsense D435 device.

## Minimum requirements
* librealsense 2.29.0
* OpenCV 4.4.0

## Preliminary operations

First, load the weight file from [hear](https://drive.google.com/file/d/107zLnzrrfdNwuWvCqO0OsIu0qvgIbWuA/view?usp=sharing), and put it in the data folder.

## How to execute

* mkdir build
* cd build
* cmake ..
* make -j\<number-of-cores+1\> (for example, make -j4)
* ./holeDetectorGray

