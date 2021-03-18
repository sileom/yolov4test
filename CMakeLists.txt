cmake_minimum_required(VERSION 2.8)
project( holeDetectorGray )
#find_package( OpenCV REQUIRED)

find_package(OpenCV 4.4.0 REQUIRED
   NO_MODULE #Should be optional, tells CMake to use config mode
   PATHS /usr/local/lib/cmake/opencv4 # Tells CMake to look here
   NO_DEFAULT_PATH #and don't look anywhere else
)

find_package(Threads REQUIRED)
find_package(realsense2 REQUIRED) 
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( holeDetectorGray object_detect.cpp obj_det.hpp)
target_link_libraries( holeDetectorGray ${OpenCV_LIBS} )
target_link_libraries(holeDetectorGray Threads::Threads)
target_link_libraries(holeDetectorGray ${realsense2_LIBRARY})
