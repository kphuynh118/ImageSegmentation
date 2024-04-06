#include <stdio.h> 
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#define MAX_INTENSITY 255
#define graylevel 256  //total number of gray levels

cv::Mat Otsu128(const cv::Mat& inputImage){ //cv::Mat& outputImage
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
    if(abs(probabilityCheck - 1.0) < tol){
        std::cout << "The sum of all probabilities is 1." << std::endl;
    }
    else {
        std::cerr << "Summation error: the sum of all probabilities is" <<probabilityCheck <<std::endl;
    }
    int optimal_threshold = 0; 
    double max_bcVariance = 0; 
    for(int t=0; t<graylevel; t++){
        double weight1 = 0; 
        double weight2 = 0;
        double mean1 = 0; 
        double mean2 = 0;
        double totalMean = 0;
        //background class C1 with gray levels[0,1,..,t]
        for(int i=0; i<=t; i++){ 
            weight1 += probability[i];
            mean1 += i*probability[i]; 
        }

        //foreground class C2 with gray levels[t+1,...,255]
        for(int i=t+1; i<graylevel; i++){ 
            weight2 += probability[i];
            mean2 += i*probability[i]; 
        }

        if(weight1 == 0){
            mean1 = 0;
        }
        else{
            mean1 = mean1/weight1; 
        }
        if(weight2 == 0){
            mean2 = 0;
        }
        else{
            mean2 = mean2/weight2; 
        }
        totalMean = weight1*mean1 + weight2*mean2; 

        double bcVariance = weight1 * std::pow(mean1-totalMean,2) + weight2 * std::pow(mean2-totalMean,2); 

        if(bcVariance > max_bcVariance){
            max_bcVariance = bcVariance;
            optimal_threshold = t; 
        }
    }
    cv::Mat segmentedOtsu128_Serial = cv::Mat::zeros(inputImage.size(),inputImage.type());
    for(int y=0; y<inputImage.rows; y++){
        for(int x=0; x<inputImage.cols; x++){
            if(inputImage.at<uchar>(y,x) >= optimal_threshold){
                segmentedOtsu128_Serial.at<uchar>(y,x) = MAX_INTENSITY; 
            }
            else{
                segmentedOtsu128_Serial.at<uchar>(y,x) = 0; 
            }
        }
    }
    return segmentedOtsu128_Serial; 
    }

int main(){
    //load an greyscale image 
    std::string imagePath = "/home/605/huynh/Project/camera128.jpg"; 
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    
    if(image.empty()) {
    std::cerr << "Error: Could not open or find the image." << std::endl;
    return -1;
    }
    cv::Mat segmentedImage = Otsu128(image);

    std::string outputPath = "/home/605/huynh/Project/segmentedOtsu128_Serial.jpg";
    if(cv::imwrite(outputPath,segmentedImage)){
        std::cout << "Segmented image is succesfully saved." << outputPath << std::endl;
    } else{
        std::cerr << "Failed to save the segmented image." << std::endl;
    }

    return 0; 
}
