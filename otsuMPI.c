#include <stdio.h> 
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <mpi.h>

//To compile: mpicxx -Wall -g -o otsuMPI otsuMPI.c `pkg-config --cflags --libs opencv`
//To execute: mpirun --oversubscribe -n [number of threads] ./otsuMPI
#define MAX_INTENSITY 255
#define GRAYLEVEL 256

bool openImage(cv::Mat &image, const std::string &imagePath){
    image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    return !image.empty();
}

std::vector<int> computeHistogram(const cv::Mat &inputImage){
    std::vector<int> histogram(GRAYLEVEL, 0);
    for(int y=0; y<inputImage.rows; y++){
        for(int x=0; x<inputImage.cols; x++){
            histogram[inputImage.at<uchar>(y,x)]++; 
        }
    }
    return histogram; 
}

std::vector<double> computeProbability(const std::vector<int> &histogram, int N){
    std::vector<double> probability(GRAYLEVEL, 0.0);
    for(int i=0; i<GRAYLEVEL; i++){
        probability[i] = static_cast<double>(histogram[i])/N; 
    }
    return probability;
}

cv::Mat getSegmentedImage(const cv::Mat& inputImage, int threshold){
    cv::Mat segmentedOtsu(inputImage.size(),inputImage.type());
    for(int y=0; y<inputImage.rows; y++){
        for(int x=0; x<inputImage.cols; x++){
            if(inputImage.at<uchar>(y,x) >= threshold){
                segmentedOtsu.at<uchar>(y,x) = MAX_INTENSITY; 
            }
            else{
                segmentedOtsu.at<uchar>(y,x) = 0; 
            }
        }
    }
    return segmentedOtsu; 
}

int main(int argc, char* argv[]){
    
    MPI_Init(&argc, &argv);
    int my_rank, comm_sz;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    //////////////// 128 //////////////////////////////
    cv::Mat image128;
    std::vector<double> probability128(GRAYLEVEL,0.0);
    std::vector<int> histogram128;

    if(my_rank==0){
        openImage(image128,"/home/605/huynh/project/camera128.jpg");
        
        if(!openImage(image128,"/home/605/huynh/project/camera128.jpg")){
            std::cerr << "Failed to open the image at " << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1); //ensure all processes not run avoiding further erroneous operations
            return 1;
        }
        
        histogram128 = computeHistogram(image128);
        probability128 = computeProbability(histogram128, image128.total());
    }

    auto start128 = std::chrono::high_resolution_clock::now(); 
    //broadcast the probability vector to all the process
    MPI_Bcast(probability128.data(), GRAYLEVEL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    //each process computes its local maximum variance and the associated threshold.
    double max_bcVariance_128 = 0;
    int optimal_threshold_128 = 0;
    for(int t = 0; t < GRAYLEVEL; t++){
        double weight1_128 = 0; 
        double weight2_128 = 0;
        double mean1_128 = 0; 
        double mean2_128 = 0;
        double totalMean_128 = 0;
        //background class C1 with gray levels[0,1,..,t]
        //printf("here1 %d of %d \n",my_rank,comm_sz);
        for(int i=0; i<=t; i++){
            weight1_128 += probability128[i];
            mean1_128 += i*probability128[i];
            
        }
        //printf("here4 %d of %d \n",my_rank,comm_sz);
        //foreground class C2 with gray levels[t+1,...,255]
        for(int i=t+1; i<GRAYLEVEL; i++){ 
            weight2_128 += probability128[i];
            mean2_128 += i*probability128[i]; 
            
        }
        
        if(weight1_128 == 0){
            mean1_128 = 0;
        }
        else{
            mean1_128 = mean1_128/weight1_128; 
        }
        if(weight2_128 == 0){
            mean2_128 = 0;
        }
        else{
            mean2_128 = mean2_128/weight2_128; 
        }
        totalMean_128 = weight1_128*mean1_128 + weight2_128*mean2_128; 
        
        //array of between class variance
        double bcVariance_128 = weight1_128 * std::pow(mean1_128-totalMean_128,2) + weight2_128 * std::pow(mean2_128-totalMean_128,2); 
        if (bcVariance_128 > max_bcVariance_128) {
            max_bcVariance_128 = bcVariance_128;
            optimal_threshold_128 = t;
        }
    }   
  
    //structure for maximum variance and its corresponding threshold
    struct { 
        double value_128;
        int index_128;
    } local_max_128 = {max_bcVariance_128, optimal_threshold_128}, global_max128;
  
    //reduce to find the global maximum variance and corresponding threshold, and this threshold is the optimal threshold
    //this method is very efficient because it consolidates all the local computations into a single global result in one step, reducing network traffic and synchronization overhead
    MPI_Reduce(&local_max_128, &global_max128, 1, MPI_DOUBLE_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
    
    //process 0 automatically receives the maximum variance and its threshold directly from the MPI_Reduce
    if(my_rank==0){
        std::cout << "Optimal threshold of the camera man image is " << global_max128.index_128  << std::endl;
        
        int threshold_128 = global_max128.index_128;
        // //int threshold = findOptThreshold(global_bcVariance);
        auto end128 = std::chrono::high_resolution_clock::now(); 
        std::chrono::duration<double> elapsed128 = end128 - start128; 
        std::cout << "Parallel code using MPI takes " << elapsed128.count() << " seconds to segment a " << image128.rows << "x" << image128.cols << " camera man image\n";
        cv::Mat segmented128 = getSegmentedImage(image128,threshold_128);
        std::string outputPath128 = "/home/605/huynh/project/cameraOtsu" + std::to_string(image128.rows) + "_MPI.jpg"; 
        cv::imwrite(outputPath128, segmented128);
        std::cout << "Segmented image save to " << outputPath128 << std::endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
     //////////////// 256 //////////////////////////////
    cv::Mat image256;
    std::vector<double> probability256(GRAYLEVEL,0.0);
    std::vector<int> histogram256;

    if(my_rank==0){
        openImage(image256,"/home/605/huynh/project/camera256.jpg");
        
        if(!openImage(image256,"/home/605/huynh/project/camera256.jpg")){
            std::cerr << "Failed to open the image at " << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1); //ensure all processes not run avoiding further erroneous operations
            return 1;
        }
        
        histogram256 = computeHistogram(image256);
        probability256 = computeProbability(histogram256, image256.total());
    }

    auto start256 = std::chrono::high_resolution_clock::now(); 
    MPI_Bcast(probability256.data(), GRAYLEVEL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    double max_bcVariance_256 = 0;
    int optimal_threshold_256 = 0;
    for(int t = 0; t < GRAYLEVEL; t++){
        double weight1_256 = 0; 
        double weight2_256 = 0;
        double mean1_256 = 0; 
        double mean2_256 = 0;
        double totalMean_256 = 0;
        //background class C1 with gray levels[0,1,..,t]
        //printf("here1 %d of %d \n",my_rank,comm_sz);
        for(int i=0; i<=t; i++){
            weight1_256 += probability256[i];
            mean1_256 += i*probability256[i];
            
        }
        //printf("here4 %d of %d \n",my_rank,comm_sz);
        //foreground class C2 with gray levels[t+1,...,255]
        for(int i=t+1; i<GRAYLEVEL; i++){ 
            weight2_256 += probability256[i];
            mean2_256 += i*probability256[i]; 
        }
        
        if(weight1_256 == 0){
            mean1_256 = 0;
        }
        else{
            mean1_256 = mean1_256/weight1_256; 
        }
        if(weight2_256 == 0){
            mean2_256 = 0;
        }
        else{
            mean2_256 = mean2_256/weight2_256; 
        }
        totalMean_256 = weight1_256*mean1_256 + weight2_256*mean2_256; 
        
        //array of between class variance
        double bcVariance_256 = weight1_256 * std::pow(mean1_256-totalMean_256,2) + weight2_256 * std::pow(mean2_256-totalMean_256,2); 
        if (bcVariance_256 > max_bcVariance_256) {
            max_bcVariance_256 = bcVariance_256;
            optimal_threshold_256 = t;
        }
    }   

    struct {
        double value_256;
        int index_256;
    } local_max_256 = {max_bcVariance_256, optimal_threshold_256}, global_max256;

    MPI_Reduce(&local_max_256, &global_max256, 1, MPI_DOUBLE_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
    if(my_rank==0){
        std::cout << "Optimal threshold of the camera man image is " << global_max256.index_256  << std::endl;
        
        int threshold_256 = global_max256.index_256;
        // //int threshold = findOptThreshold(global_bcVariance);
        auto end256 = std::chrono::high_resolution_clock::now(); 
        std::chrono::duration<double> elapsed256 = end256 - start256; 
        std::cout << "Parallel code using MPI takes " << elapsed256.count() << " seconds to segment a " << image256.rows << "x" << image256.cols << " camera man image\n";
        cv::Mat segmented256 = getSegmentedImage(image256,threshold_256);
        std::string outputPath256 = "/home/605/huynh/project/cameraOtsu" + std::to_string(image256.rows) + "_MPI.jpg"; 
        cv::imwrite(outputPath256, segmented256);
        std::cout << "Segmented image save to " << outputPath256 << std::endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
     //////////////// 512 //////////////////////////////
    cv::Mat image512;
    std::vector<double> probability512(GRAYLEVEL,0.0);
    std::vector<int> histogram512;

    if(my_rank==0){
        openImage(image512,"/home/605/huynh/project/camera512.jpg");
        
        if(!openImage(image512,"/home/605/huynh/project/camera512.jpg")){
            std::cerr << "Failed to open the image at " << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1); //ensure all processes not run avoiding further erroneous operations
            return 1;
        }
        
        histogram512 = computeHistogram(image512);
        probability512 = computeProbability(histogram512, image512.total());
    }

    auto start512 = std::chrono::high_resolution_clock::now(); 
    MPI_Bcast(probability512.data(), GRAYLEVEL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    double max_bcVariance_512 = 0;
    int optimal_threshold_512 = 0;
    for(int t = 0; t < GRAYLEVEL; t++){
        double weight1_512 = 0; 
        double weight2_512 = 0;
        double mean1_512 = 0; 
        double mean2_512 = 0;
        double totalMean_512 = 0;
        //background class C1 with gray levels[0,1,..,t]
        //printf("here1 %d of %d \n",my_rank,comm_sz);
        for(int i=0; i<=t; i++){
            weight1_512 += probability512[i];
            mean1_512 += i*probability512[i];
            
        }
        //printf("here4 %d of %d \n",my_rank,comm_sz);
        //foreground class C2 with gray levels[t+1,...,255]
        for(int i=t+1; i<GRAYLEVEL; i++){ 
            weight2_512 += probability512[i];
            mean2_512 += i*probability512[i]; 
        }
        
        if(weight1_512 == 0){
            mean1_512 = 0;
        }
        else{
            mean1_512 = mean1_512/weight1_512; 
        }
        if(weight2_512 == 0){
            mean2_512 = 0;
        }
        else{
            mean2_512 = mean2_512/weight2_512; 
        }
        totalMean_512 = weight1_512*mean1_512 + weight2_512*mean2_512; 
        
        //array of between class variance
        double bcVariance_512 = weight1_512 * std::pow(mean1_512-totalMean_512,2) + weight2_512 * std::pow(mean2_512-totalMean_512,2); 
        if (bcVariance_512 > max_bcVariance_512) {
            max_bcVariance_512 = bcVariance_512;
            optimal_threshold_512 = t;
        }
    }   
    
    struct {
        double value_512;
        int index_512;
    } local_max_512 = {max_bcVariance_512, optimal_threshold_512}, global_max512;

    MPI_Reduce(&local_max_512, &global_max512, 1, MPI_DOUBLE_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
    if(my_rank==0){
        std::cout << "Optimal threshold of the camera man image is " << global_max512.index_512  << std::endl;
        
        int threshold_512 = global_max512.index_512;
        // //int threshold = findOptThreshold(global_bcVariance);
        auto end512 = std::chrono::high_resolution_clock::now(); 
        std::chrono::duration<double> elapsed512 = end512 - start512; 
        std::cout << "Parallel code using MPI takes " << elapsed512.count() << " seconds to segment a " << image512.rows << "x" << image512.cols << " camera man image\n";
        cv::Mat segmented512 = getSegmentedImage(image512,threshold_512);
        std::string outputPath512 = "/home/605/huynh/project/cameraOtsu" + std::to_string(image512.rows) + "_MPI.jpg"; 
        cv::imwrite(outputPath512, segmented512);
        std::cout << "Segmented image save to " << outputPath512 << std::endl;
    }
 
    MPI_Finalize();
    return 0;
} 
