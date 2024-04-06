#include <stdio.h> 
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#define MAX_INTENSITY 255
const int graylevel = 256;  //total number of gray levels

void Otsu128(const cv::Mat& inputImage){ //cv::Mat& outputImage
    int N = 16384; //total number of pixels = number of  cols x number of rows
    std::vector<int> histogram(graylevel, 0); //store the histogram with a size of 256 for all possible gray levels 0-255
    for(int y=0; y<inputImage.rows; y++){
        for(int x=0; x<inputImage.cols; x++){
            histogram[inputImage.at<uchar>(y,x)]++; 
        }
    }
    std::vector<double> probability(graylevel,0.0);
    for(int i=0; i<graylevel; i++){
        probability[i] = (double) histogram[i] / N; 
    }

    double probabilityCheck = 0.0;
    for(double p : probability){
        probabilityCheck += p; 
    }
    const double tol = 1e-6;
    if(std::abs(probabilityCheck - 1.0) < tol){
        std::cout << "The sum of all probabilities is 1." << std::endl;
    }
    else {
        std::cerr << "Summation error: the sum of all probabilities is" <<probabilityCheck <<std::endl;
    }
}

int main(){
    //load an greyscale image 
    std::string imagePath = "/home/605/huynh/Project/camera128.jpg"; 
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    
    if (image.empty()) {
    std::cerr << "Error: Could not open or find the image." << std::endl;
    return -1;
    }
    Otsu128(image);


    return 0; 
}
