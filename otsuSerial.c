#include <stdio.h> 
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>

#define MAX_INTENSITY 255
#define GRAYLEVEL 256  //total number of gray levels

cv::Mat Otsu128(const cv::Mat& inputImage){ //cv::Mat& outputImage
    int N = 128*128; //total number of pixels = number of  cols x number of rows
    std::vector<int> histogram(GRAYLEVEL, 0); //store the histogram with a size of 256 for all possible gray levels 0-255
    for(int y=0; y<inputImage.rows; y++){
        for(int x=0; x<inputImage.cols; x++){
            histogram[inputImage.at<uchar>(y,x)]++; 
        }
    }
    std::vector<double> probability(GRAYLEVEL,0.0);
    for(int i=0; i<GRAYLEVEL; i++){
        probability[i] = (double) histogram[i] / N; 
    }

    int optimal_threshold = 0; 
    double max_bcVariance = 0; 
    for(int t=0; t<GRAYLEVEL; t++){
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
        for(int i=t+1; i<GRAYLEVEL; i++){ 
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
        
        //between class variance
        double bcVariance = weight1 * std::pow(mean1-totalMean,2) + weight2 * std::pow(mean2-totalMean,2); 
        
        //the maximum between class variance = the minumum within class variance = optimal threshold 
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
cv::Mat Otsu256(const cv::Mat& inputImage){ //cv::Mat& outputImage
    int N = 256*256; //total number of pixels = number of  cols x number of rows
    std::vector<int> histogram(GRAYLEVEL, 0); //store the histogram with a size of 256 for all possible gray levels 0-255
    for(int y=0; y<inputImage.rows; y++){
        for(int x=0; x<inputImage.cols; x++){
            histogram[inputImage.at<uchar>(y,x)]++; 
        }
    }
    std::vector<double> probability(GRAYLEVEL,0.0);
    for(int i=0; i<GRAYLEVEL; i++){
        probability[i] = (double) histogram[i] / N; 
    }

    int optimal_threshold = 0; 
    double max_bcVariance = 0; 
    for(int t=0; t<GRAYLEVEL; t++){
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
        for(int i=t+1; i<GRAYLEVEL; i++){ 
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
        
        //between class variance
        double bcVariance = weight1 * std::pow(mean1-totalMean,2) + weight2 * std::pow(mean2-totalMean,2); 
        
        //the maximum between class variance = the minumum within class variance = optimal threshold 
        if(bcVariance > max_bcVariance){
            max_bcVariance = bcVariance;
            optimal_threshold = t; 
        }
    }
    cv::Mat segmentedOtsu256_Serial = cv::Mat::zeros(inputImage.size(),inputImage.type());
    for(int y=0; y<inputImage.rows; y++){
        for(int x=0; x<inputImage.cols; x++){
            if(inputImage.at<uchar>(y,x) >= optimal_threshold){
                segmentedOtsu256_Serial.at<uchar>(y,x) = MAX_INTENSITY; 
            }
            else{
                segmentedOtsu256_Serial.at<uchar>(y,x) = 0; 
            }
        }
    }
    return segmentedOtsu256_Serial; 
}

cv::Mat Otsu512(const cv::Mat& inputImage){ //cv::Mat& outputImage
    int N = 512*512; //total number of pixels = number of  cols x number of rows
    std::vector<int> histogram(GRAYLEVEL, 0); //store the histogram with a size of 256 for all possible gray levels 0-255
    for(int y=0; y<inputImage.rows; y++){
        for(int x=0; x<inputImage.cols; x++){
            histogram[inputImage.at<uchar>(y,x)]++; 
        }
    }
    std::vector<double> probability(GRAYLEVEL,0.0);
    for(int i=0; i<GRAYLEVEL; i++){
        probability[i] = (double) histogram[i] / N; 
    }

    int optimal_threshold = 0; 
    double max_bcVariance = 0; 
    for(int t=0; t<GRAYLEVEL; t++){
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
        for(int i=t+1; i<GRAYLEVEL; i++){ 
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
        
        //between class variance
        double bcVariance = weight1 * std::pow(mean1-totalMean,2) + weight2 * std::pow(mean2-totalMean,2); 
        
        //the maximum between class variance = the minumum within class variance = optimal threshold 
        if(bcVariance > max_bcVariance){
            max_bcVariance = bcVariance;
            optimal_threshold = t; 
        }
    }
    cv::Mat segmentedOtsu512_Serial = cv::Mat::zeros(inputImage.size(),inputImage.type());
    for(int y=0; y<inputImage.rows; y++){
        for(int x=0; x<inputImage.cols; x++){
            if(inputImage.at<uchar>(y,x) >= optimal_threshold){
                segmentedOtsu512_Serial.at<uchar>(y,x) = MAX_INTENSITY; 
            }
            else{
                segmentedOtsu512_Serial.at<uchar>(y,x) = 0; 
            }
        }
    }
    return segmentedOtsu512_Serial; 
}

int main(){
    //load an greyscale image 
    std::string imagePath128 = "/home/605/huynh/Project/camera128.jpg"; 
    cv::Mat image128 = cv::imread(imagePath128, cv::IMREAD_GRAYSCALE);
    
    if(image128.empty()) {
    std::cerr << "Error: Could not open or find the image." << std::endl;
    return -1;
    }
    
    auto start128 = std::chrono::high_resolution_clock::now(); 
    cv::Mat segmentedImage128 = Otsu128(image128);
    auto end128 = std::chrono::high_resolution_clock::now(); 

    std::chrono::duration<double> elapsed128 = end128 - start128;
    std::cout << "Serial code takes " << elapsed128.count() << " seconds to segment a 128x128 camera man image\n";

    std::string outputPath128 = "/home/605/huynh/Project/segmentedOtsu128_Serial.jpg";
    if(cv::imwrite(outputPath128,segmentedImage128)){
        std::cout << "Segmented 128x128 image is succesfully saved to " << outputPath128 << std::endl;
    } else{
        std::cerr << "Failed to save the segmented image." << std::endl;
    }

    printf("\n");
    std::string imagePath256 = "/home/605/huynh/Project/camera256.jpg"; 
    cv::Mat image256 = cv::imread(imagePath256, cv::IMREAD_GRAYSCALE);
    
    if(image256.empty()) {
    std::cerr << "Error: Could not open or find the image." << std::endl;
    return -1;
    }
    
    auto start256 = std::chrono::high_resolution_clock::now(); 
    cv::Mat segmentedImage256 = Otsu256(image256);
    auto end256 = std::chrono::high_resolution_clock::now(); 

    std::chrono::duration<double> elapsed256 = end256 - start256;
    std::cout << "Serial code takes " << elapsed256.count() << " seconds to segment a 256x256 camera man image\n";

    std::string outputPath256 = "/home/605/huynh/Project/segmentedOtsu256_Serial.jpg";
    if(cv::imwrite(outputPath256,segmentedImage256)){
        std::cout << "Segmented 256x256 image is succesfully saved to " << outputPath256 << std::endl;
    } else{
        std::cerr << "Failed to save the segmented image." << std::endl;
    }

    printf("\n");
    std::string imagePath512 = "/home/605/huynh/Project/camera512.jpg"; 
    cv::Mat image512 = cv::imread(imagePath512, cv::IMREAD_GRAYSCALE);
    
    if(image512.empty()) {
    std::cerr << "Error: Could not open or find the image." << std::endl;
    return -1;
    }
    
    auto start512 = std::chrono::high_resolution_clock::now(); 
    cv::Mat segmentedImage512 = Otsu512(image512);
    auto end512 = std::chrono::high_resolution_clock::now(); 

    std::chrono::duration<double> elapsed512 = end512 - start512;
    std::cout << "Serial code takes " << elapsed512.count() << " seconds to segment a 512x512 camera man image\n";

    std::string outputPath512 = "/home/605/huynh/Project/segmentedOtsu512_Serial.jpg";
    if(cv::imwrite(outputPath512,segmentedImage512)){
        std::cout << "Segmented 512x512 image is succesfully saved to " << outputPath512 << std::endl;
    } else{
        std::cerr << "Failed to save the segmented image." << std::endl;
    }
    
    return 0; 
}
