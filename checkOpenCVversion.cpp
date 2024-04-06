#include <opencv2/core/core.hpp>
#include <iostream>
//to compile: g++ checkOpenCVversion.cpp -o checkOpenCVversion `pkg-config --cflags --libs opencv`
int main() {
    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;
    return 0;
}
