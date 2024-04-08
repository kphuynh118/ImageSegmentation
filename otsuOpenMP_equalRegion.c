#include <stdio.h> 
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

#define MAX_INTENSITY 255
#define GRAYLEVEL 256
#define REGION_SIZE 16 

int openImage(cv::Mat &image, std::string imagePath){

    image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    
    if (image.empty()) {
    std::cerr << "Error: Could not open or find the image." << std::endl;
        return 0; 
    }
    return 0;
}

cv::Mat OtsuOpenMP_equalRegion(const cv::Mat &inputImage){

    int rows = inputImage.rows;
    int cols = inputImage.cols;
    int N = rows*cols; 
    //Calculate the number of regions
    int nRegionsX = (cols+REGION_SIZE-1)/REGION_SIZE;
    int nRegionsY = (rows+REGION_SIZE-1)/REGION_SIZE;
    int totalRegions = nRegionsX*nRegionsY; 

    omp_set_num_threads(totalRegions); //number of threads is the number of regions

    std::vector<int> histogram(GRAYLEVEL, 0); 
    //Compute a histogram
    for(int y=0; y<inputImage.rows; y++){
        for(int x=0; x<inputImage.cols; x++){
            histogram[inputImage.at<uchar>(y,x)]++; 
        }
    }
    
    std::vector<double> probability(GRAYLEVEL,0.0); //vector that stores the probability of each intensity level
    for(int i=0; i<GRAYLEVEL; i++){
        probability[i] = (double) histogram[i] / N; 
    }

    double max_bcVariance = 0; 
    double bcVariance[GRAYLEVEL];

    #pragma omp parallel for reduction(max:max_bcVariance)
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
        
        //array of between class variance
        bcVariance[t] = weight1 * std::pow(mean1-totalMean,2) + weight2 * std::pow(mean2-totalMean,2); 
    }
    
    //Find the maximum between class variance after the parallel part, ensure that optimal threshold value is the same as the serial version
    //the maximum between class variance = the minumum within class variance = optimal threshold 
    int optimal_threshold = 0; 
    for(int t=0; t<GRAYLEVEL; t++){
        if(bcVariance[t] > max_bcVariance){
            max_bcVariance = bcVariance[t];
            optimal_threshold = t;   
        }   
    }    

    printf("Optimal threshold of a %dx%d camera man image is %d\n",rows, cols, optimal_threshold);
    
    cv::Mat segmentedOtsu_OpenMP = cv::Mat::zeros(inputImage.size(),inputImage.type());
    for(int y=0; y<rows; y++){
        for(int x=0; x<cols; x++){
            if(inputImage.at<uchar>(y,x) >= optimal_threshold){
                segmentedOtsu_OpenMP.at<uchar>(y,x) = MAX_INTENSITY; 
            }
            else{
                segmentedOtsu_OpenMP.at<uchar>(y,x) = 0; 
            }
        }
    }
    return segmentedOtsu_OpenMP; 
}

int main(){
    
    cv::Mat image128, image256, image512; 

    /////////////////////////////128////////////////////////////////
    openImage(image128,"/home/605/huynh/Project/camera128.jpg"); 
    auto start128 = std::chrono::high_resolution_clock::now(); 
    cv::Mat segmentedImageOMP128 = OtsuOpenMP_equalRegion(image128);
    auto end128 = std::chrono::high_resolution_clock::now(); 

    std::chrono::duration<double> elapsed128 = end128 - start128;
    std::cout << "Parallel code using OpenMP takes " << elapsed128.count() << " seconds to segment a 128x128 camera man image\n";

    std::string outputPath128 = "/home/605/huynh/Project/cameraOtsu128_OpenMP_equalRegion.jpg";
    if(cv::imwrite(outputPath128,segmentedImageOMP128)){
        std::cout << "Segmented 128x128 image using OpenMP is succesfully saved to " << outputPath128 << std::endl;
    } else{
        std::cerr << "Failed to save the segmented image." << std::endl;
    }
    printf("\n");
    /////////////////////////////256////////////////////////////////
    openImage(image256,"/home/605/huynh/Project/camera256.jpg"); 
    auto start256 = std::chrono::high_resolution_clock::now(); 
    cv::Mat segmentedImageOMP256 = OtsuOpenMP_equalRegion(image256);
    auto end256 = std::chrono::high_resolution_clock::now(); 

    std::chrono::duration<double> elapsed256 = end256 - start256;
    std::cout << "Parallel code using OpenMP takes " << elapsed256.count() << " seconds to segment a 256x256 camera man image\n";

    std::string outputPath256 = "/home/605/huynh/Project/cameraOtsu256_OpenMP_equalRegion.jpg";
    if(cv::imwrite(outputPath256,segmentedImageOMP256)){
        std::cout << "Segmented 256x256 image using OpenMP is succesfully saved to " << outputPath256 << std::endl;
    } else{
        std::cerr << "Failed to save the segmented image." << std::endl;
    }
    printf("\n");
    /////////////////////////////512////////////////////////////////
    openImage(image512,"/home/605/huynh/Project/camera512.jpg"); 
    auto start512 = std::chrono::high_resolution_clock::now(); 
    cv::Mat segmentedImageOMP512 = OtsuOpenMP_equalRegion(image512);
    auto end512 = std::chrono::high_resolution_clock::now(); 

    std::chrono::duration<double> elapsed512 = end512 - start512;
    std::cout << "Parallel code using OpenMP takes " << elapsed512.count() << " seconds to segment a 512x512 camera man image\n";

    std::string outputPath512 = "/home/605/huynh/Project/cameraOtsu512_OpenMP_equalRegion.jpg";
    if(cv::imwrite(outputPath512,segmentedImageOMP512)){
        std::cout << "Segmented 512x512 image using OpenMP is succesfully saved to " << outputPath512 << std::endl;
    } else{
        std::cerr << "Failed to save the segmented image." << std::endl;
    }

    return 0;
} 
