
#include "obj_det.hpp"

int main(int argc, char** argv)
{
    std::cout << "Hole detection realsense" << std::endl;
    Mat result_l;
    Mat result_r;
    Net net = loadNet();
    pipeline pipe = startCamera();
    detect(false, result_l, result_r, net, pipe);
    //cv::imwrite("result/LEFT.png", result_l);
    //cv::imwrite("result/RIGHT.png", result_r);
    return 0;
}
