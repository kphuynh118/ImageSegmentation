#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

int openImage(cv::Mat &image, std::string imagePath);
void kmeansMPI(uchar* image, uchar* original, int myRank, int threadCount, int size);
void recolor(int myRank, int threadCount, uchar* image, uchar* original, int size);
void recenter(int myRank, int threadCount);
int color1 = 0, color2 = 0, c1Sum = 0, c2Sum = 0, c1, c2, iter;
// auto start, end; 
cv::Mat image128, original128, image256, original256, image512, original512;


int main(int argc, char *argv[]){
  
    MPI_Init(&argc, &argv); 
    int myRank; 
    int threadCount;
    // c1Sum = 0; 
    // c2Sum = 0; 
    // color1 = 0; 
    // color2 = 0; 
    // iter = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &threadCount);
    int size;
    int rows;
//////////////////128///////////////////////////
    if (myRank == 0){
            openImage(image128, "/home/605/ulloa/project/camera128.jpg");
            //image128.copyTo(original128);
            rows = image128.rows;
            // openImage(image256, "/home/605/./ulloa/project/camera256.jpg");
            // image256.copyTo(original256);
            // openImage(image512, "/home/605/ulloa/project/camera512.jpg");
            // image512.copyTo(original512); 
    }
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(myRank < rows%threadCount){
        size = (rows/threadCount + 1)*rows;
    }
    else{
        size = (rows/threadCount)*rows;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    uchar* partImg = new uchar[size];
    uchar* partOrig = new uchar[size];
    MPI_Scatter(image128.data, size, MPI_UNSIGNED_CHAR, partImg, size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(image128.data, size, MPI_UNSIGNED_CHAR, partOrig, size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start1 = std::chrono::high_resolution_clock::now(); 
    if (myRank == 0){
        c1 = (rand() % 256);
        c2 = (rand() % 256);
        iter = 0; 
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&c1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&c2, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    while (iter < 1000){
        kmeansMPI(partImg, partOrig, myRank, threadCount, size);
        MPI_Barrier(MPI_COMM_WORLD);
        recenter(myRank, threadCount);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (myRank == 0){
        if (color1 < color2){
            color1 = 0; 
            color2 = 255; 
        }
        else{ 
            color1 = 255; 
            color2 = 0; 
        }
    }
    MPI_Bcast(&color1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&color2, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    recolor(myRank, threadCount, partImg, partOrig, size);
    MPI_Barrier(MPI_COMM_WORLD);
    auto end1 = std::chrono::high_resolution_clock::now(); 
    std::chrono::duration<double> elapsed1 = end1 - start1;
    cv::Mat outImage;
    if (myRank == 0 ){ 
        outImage = cv::Mat(image128.size(), image128.type() );
    }
    MPI_Gather(partImg, size, MPI_UNSIGNED_CHAR, outImage.data, size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    if (myRank == 0){
        std::cout << "MPI code takes " << elapsed1.count() << " seconds to segment a "<< image128.rows <<"x" << image128.cols <<" camera man image\n";
        std::string outputPath128 = "/home/605/ulloa/project/camera128kMeans_MPI.jpg";
            if(cv::imwrite(outputPath128,outImage)){
                std::cout << "Segmented 128x128 image is succesfully saved to " << outputPath128 << "\n"<<std::endl;
            } else{
                std::cerr << "Failed to save the segmented image.\n" << std::endl;
            }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // ///////////// 256 //////////
    if (myRank == 0){
            openImage(image256, "/home/605/ulloa/project/camera256.jpg");
            rows = image256.rows; 
    }
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(myRank < rows%threadCount){
        size = (rows/threadCount + 1)*rows;
    }
    else{
        size = (rows/threadCount)*rows;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    uchar* partImg2 = new uchar[size];
    uchar* partOrig2 = new uchar[size];
    MPI_Scatter(image256.data, size, MPI_UNSIGNED_CHAR, partImg2, size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(image256.data, size, MPI_UNSIGNED_CHAR, partOrig2, size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start2 = std::chrono::high_resolution_clock::now(); 
    if (myRank == 0){
        c1 = (rand() % 256);
        c2 = (rand() % 256);
        iter = 0; 
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&c1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&c2, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    while (iter < 1000){
        kmeansMPI(partImg2, partOrig2, myRank, threadCount, size);
        MPI_Barrier(MPI_COMM_WORLD);
        recenter(myRank, threadCount);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (myRank == 0){
        if (color1 < color2){
            color1 = 0; 
            color2 = 255; 
        }
        else{ 
            color1 = 255; 
            color2 = 0; 
        }
    }
    MPI_Bcast(&color1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&color2, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    recolor(myRank, threadCount, partImg2, partOrig2, size);
    MPI_Barrier(MPI_COMM_WORLD);
    auto end2 = std::chrono::high_resolution_clock::now(); 
    std::chrono::duration<double> elapsed2 = end2 - start2;
    cv::Mat outImage2;
    if (myRank == 0 ){ 
        outImage2 = cv::Mat(image256.size(), image256.type() );
    }
    MPI_Gather(partImg2, size, MPI_UNSIGNED_CHAR, outImage2.data, size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    if (myRank == 0){
        std::cout << "MPI code takes " << elapsed2.count() << " seconds to segment a "<< image256.rows <<"x" << image256.cols <<" camera man image\n";
        std::string outputPath256 = "/home/605/ulloa/project/camera256kMeans_MPI.jpg";
            if(cv::imwrite(outputPath256,outImage2)){
                std::cout << "Segmented 256x256 image is succesfully saved to " << outputPath256 << "\n"<<std::endl;
            } else{
                std::cerr << "Failed to save the segmented image.\n" << std::endl;
            }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // //////////// 512 ////////////
        if (myRank == 0){
            openImage(image512, "/home/605/ulloa/project/camera512.jpg");
            rows = image512.rows; 
    }
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(myRank < rows%threadCount){
        size = (rows/threadCount + 1)*rows;
    }
    else{
        size = (rows/threadCount)*rows;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    uchar* partImg3 = new uchar[size];
    uchar* partOrig3 = new uchar[size];
    MPI_Scatter(image512.data, size, MPI_UNSIGNED_CHAR, partImg3, size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(image512.data, size, MPI_UNSIGNED_CHAR, partOrig3, size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start3 = std::chrono::high_resolution_clock::now(); 
    if (myRank == 0){
        c1 = (rand() % 256);
        c2 = (rand() % 256);
        iter = 0;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&c1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&c2, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    while (iter < 1000){
        kmeansMPI(partImg3, partOrig3, myRank, threadCount, size);
        MPI_Barrier(MPI_COMM_WORLD);
        recenter(myRank, threadCount);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (myRank == 0){
        if (color1 < color2){
            color1 = 0; 
            color2 = 255; 
        }
        else{ 
            color1 = 255; 
            color2 = 0; 
        }
    }
    MPI_Bcast(&color1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&color2, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    recolor(myRank, threadCount, partImg3, partOrig3, size);
    MPI_Barrier(MPI_COMM_WORLD);
    auto end3 = std::chrono::high_resolution_clock::now(); 
    std::chrono::duration<double> elapsed3 = end3 - start3;
    cv::Mat outImage3;
    if (myRank == 0 ){ 
        outImage3 = cv::Mat(image512.size(), image512.type() );
    }
    MPI_Gather(partImg3, size, MPI_UNSIGNED_CHAR, outImage3.data, size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    if (myRank == 0){
        std::cout << "MPI code takes " << elapsed3.count() << " seconds to segment a "<< image512.rows <<"x" << image512.cols <<" camera man image\n";
        std::string outputPath512 = "/home/605/ulloa/project/camera512kMeans_MPI.jpg";
            if(cv::imwrite(outputPath512,outImage3)){
                std::cout << "Segmented 512x512 image is succesfully saved to " << outputPath512 << "\n"<<std::endl;
            } else{
                std::cerr << "Failed to save the segmented image.\n" << std::endl;
            }
    }
    ///////////////////////////////
    MPI_Finalize();
    return 0; 
}

int openImage(cv::Mat &image, std::string imagePath){

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

void kmeansMPI(uchar* image, uchar* original, int myRank, int threadCount, int size){ 
    int i;
    int localC1Sum=0, localC2Sum=0, localColor1=0, localColor2=0;
    int d1, d2;  
    int val;
    for (i = 0; i < size; i++){
        val = original[i];
        d1 = abs(val - c1);
        d2 = abs(val - c2);
        if (d1 < d2){
            localC1Sum++; 
            localColor1 += val;
            image[i]= 1; 
        }
        else{
            localC2Sum++; 
            localColor2 += val;
            image[i] = 2; 
        }
    }
        MPI_Allreduce(&localC1Sum, &c1Sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD); 
        MPI_Allreduce(&localC2Sum, &c2Sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD); 
        MPI_Allreduce(&localColor1, &color1, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD); 
        MPI_Allreduce(&localColor2, &color2, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD); 
}

void recenter(int myRank, int threadCount){ 
    if(myRank == 0){
        color1 = color1/c1Sum; 
        color2 = color2/c2Sum; 

        int d1 = abs(color1 - c1);
        int d2 = abs(color2 - c2);
        if ((d1+d2)/2 < 1e-6){
            iter = 1000; 
        }
        else{
            c1 = color1; 
            c2= color2; 
            iter += 1;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&c1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&c2, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return;
}

void recolor(int myRank, int threadCount, uchar* image, uchar* original, int size){
    int i; 
    for (i = 0; i < size; i++){
            if (image[i] == 1){
                image[i] = color1; 
            }
            else{
                image[i] = color2; 
            }
    }
}