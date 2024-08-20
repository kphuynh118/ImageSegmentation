#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

int openImage(cv::Mat &image, std::string imagePath);
int kmeansOp(cv::Mat &image, cv::Mat &original);
int threadCount; 

int main(int argc, char *argv[]){
    threadCount = strtol(argv[1], NULL, 10);
    
     cv::Mat image128, original128, image256, original256, image512, original512; 
    /////////// 128 /////////////
    openImage(image128, "/home/605/ulloa/project/camera128.jpg");
    image128.copyTo(original128);
    kmeansOp(image128, original128);
    std::string outputPath128 = "/home/605/ulloa/project/camera128kMeans_OP.jpg";
    if(cv::imwrite(outputPath128,image128)){
        std::cout << "Segmented 128x128 image is succesfully saved to " << outputPath128 << "\n"<<std::endl;
    } else{
        std::cerr << "Failed to save the segmented image.\n" << std::endl;
    }
    
    ///////////// 256 //////////
    openImage(image256, "/home/605/ulloa/project/camera256.jpg");
    image256.copyTo(original256);
    kmeansOp(image256, original256);

    std::string outputPath256 = "/home/605/ulloa/project/camera256kMeans_OP.jpg";
    if(cv::imwrite(outputPath256,image256)){
        std::cout << "Segmented 256x256 image is succesfully saved to " << outputPath256 << "\n"<<std::endl;
    } else{
        std::cerr << "Failed to save the segmented image.\n" << std::endl;
    }
    //////////// 512 ////////////
    openImage(image512, "/home/605/ulloa/project/camera512.jpg");
    image512.copyTo(original512);
    kmeansOp(image512, original512);

    std::string outputPath512 = "/home/605/ulloa/project/camera512kMeans_OP.jpg";
    if(cv::imwrite(outputPath512,image512)){
        std::cout << "Segmented 512x512 image is succesfully saved to " << outputPath512 << "\n"<<std::endl;
    } else{
        std::cerr << "Failed to save the segmented image.\n" << std::endl;
    }
    return 0; 
}

int openImage(cv::Mat &image, std::string imagePath){

    //std::string imagePath = "/home/605/ulloa/project/camera128.jpg"; 
    image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    
    if (image.empty()) {
    std::cerr << "Error: Could not open or find the image." << std::endl;
        return 0; 
    }
    //check and print the dimensions of the image
    std::cout << "Image loaded successfully!" << std::endl;
    std::cout << "Image size: " << image.cols << " x " << image.rows << std::endl;
    return 0; 
}

int kmeansOp(cv::Mat &image, cv::Mat &original){  
    auto start = std::chrono::high_resolution_clock::now(); 
    int rows = image.rows; 
    int cols = image.cols; 
    int color1, color2;
    int iter = 0; 
    int c1 = (rand() % 256);
    int c2 = (rand() % 256);
    double d1 = 0, d2 = 0;
    int c1Sum, c2Sum ;
    int i,j; 
    while (iter < 1000){
        c1Sum = 0, c2Sum = 0; 
        color1 = 0, color2 = 0; 
        #pragma omp parallel for num_threads(threadCount)\
            reduction(+:c1Sum,c2Sum,color1,color2) private(i,j,d1,d2) shared(c1, c2, cols)
        for (i = 0; i < rows; i++){
            for (j = 0; j < cols; j++){
                d1 = abs(original.at<uchar>(i, j) - c1);
                d2 = abs(original.at<uchar>(i, j) - c2);
                if (d1 < d2){
                    c1Sum++; 
                    color1 += original.at<uchar>(i, j);
                    image.at<uchar>(i, j) = 1; 
                }
                else{
                    //#pragma omp critical
                    c2Sum++; 
                    //#pragma omp critical
                    color2 += original.at<uchar>(i, j);
                    image.at<uchar>(i, j) = 2; 
                }
            }
        }
        color1 = color1/c1Sum; 
        color2 = color2/c2Sum; 
        d1 = abs(color1 - c1);
        d2 = abs(color2 - c2);
        if ((d1+d2)/2 < 1e-6){
            iter = 1000; 
        }
        else{
            c1 = color1; 
            c2= color2; 
            iter++; 
        }
    }

    if (color1 < color2){
        color1 = 0; 
        color2 = 255; 
    }
    else{ 
        color1 = 255; 
        color2 = 0; 
    }
    #pragma omp parallel for num_threads(threadCount)\
        private(i,j)
    for(i = 0; i < rows; i++){
        for(j = 0; j <cols; j++){
            if (image.at<uchar>(i, j) == 1){
                image.at<uchar>(i, j) = color1; 
            }
            else{
                image.at<uchar>(i, j) = color2; 
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now(); 

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "OpenMp code takes " << elapsed.count() << " seconds to segment a "<< rows <<"x" << cols <<" camera man image\n";
    return 0; 
}
