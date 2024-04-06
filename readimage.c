#include <stdio.h> 
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <iostream>

//OpenCV Version: 3.4.6
//To compile: g++ -Wall -g -o readimage readimage.c `pkg-config --cflags --libs opencv`
//then ./readimage
//it should print out 5x5 matrix of the top-left
int main(){

    std::string imagePath = "/home/605/huynh/Project/camera128.jpg"; 
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    
    if (image.empty()) {
    std::cerr << "Error: Could not open or find the image." << std::endl;
    return -1;
    }
    //check and print the dimensions of the image
    std::cout << "Image loaded successfully!" << std::endl;
    std::cout << "Image size: " << image.cols << " x " << image.rows << std::endl;
    std::cout << "Number of channels: " << image.channels() << std::endl; //1 for grayscale
    
    // Print out number of rows and columns
    printf("number of rows %d\n",image.rows);
    printf("number of columns %d\n",image.cols);
    
    // Print a portion of the matrix (e.g., top-left 5x5 pixels)
    int rowsToPrint = std::min(5, image.rows);
    int colsToPrint = std::min(5, image.cols);

    std::cout << "Top-left corner of the image matrix (5x5 pixels):" << std::endl;
    for (int i = 0; i < rowsToPrint; ++i) {
        for (int j = 0; j < colsToPrint; ++j) {
            std::cout << static_cast<int>(image.at<uchar>(i, j)) << " ";
        }
        std::cout << std::endl;
    }

    return 0; 
}
