# yolov4test

## Minimum requirements
* librealsense 2.29.0
* OpenCV 4.4.0

## Preliminary operations

First, load the weight file from [hear](https://drive.google.com/drive/folders/1XYIXBgzzLha0lCn7QxqOpQbrOh4AdZ8Z?usp=sharing), and put it in the data folder.

## How to execute

* mkdir build
* cd build
* cmake ..
* make -j\<number-of-cores+1\> (for example, make -j4)
* ./holeDetectorGray

