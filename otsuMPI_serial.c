#include <stdio.h> 
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <mpi.h>

//To compile: mpicxx -Wall -g -o otsuMPI otsuMPI.c `pkg-config --cflags --libs opencv`
//then ./otsuMPI -n [number of threads]
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

int findOptThreshold(const std::vector<double> &bcVariance){
    //Find the maximum between class variance after the parallel part, ensure that optimal threshold value is the same as the serial version
    //the maximum between class variance = the minumum within class variance = optimal threshold 
    double max_bcVariance = 0; 
    int optimal_threshold = 0; 
    for(int t=0; t<GRAYLEVEL; t++){
        if(bcVariance[t] > max_bcVariance){
            max_bcVariance = bcVariance[t];
            optimal_threshold = t;   
        }   
    }
    std::cout << "Optimal threshold of the camera man image is " << optimal_threshold  << std::endl;
    return optimal_threshold;    
}

int OtsuVarianceMPI(const std::vector<double> &probability, int my_rank, int comm_sz){
    //the goal is to distribute the workload evenly across processes 
    //the remainder (if any) needs to be handled efficiently
    int workload = GRAYLEVEL / comm_sz; 
    int remainder = GRAYLEVEL % comm_sz; 

    int t_start, t_end; 
    //for each process, it starts from my_rank*workload, if there's remainder, the processes will get one extra item 
    //it ends by adding t_start and workload, If the process is one of the first remainder processes, it will take an additional item 
    if(my_rank < remainder){
        t_start = my_rank * workload  + 1;
        t_end = t_start + workload + 1;
    } else{
        t_start = my_rank * workload;
        t_end = t_start + workload; 
    }

    std::vector<double> local_bcVariance(GRAYLEVEL, 0.0);
   
    for(int t = t_start; t < t_end; t++){
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
        local_bcVariance[t] = weight1 * (mean1-totalMean) * (mean1-totalMean) + weight2 * (mean2-totalMean) * (mean2-totalMean); 
    }
    
    //get all local variances from all processes
    std::vector<double> global_bcVariance(GRAYLEVEL, 0.0);
    MPI_Reduce(local_bcVariance.data(), global_bcVariance.data(), GRAYLEVEL, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    return findOptThreshold(global_bcVariance);

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

int main(int argc, char** argv){
    
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<std::string> imagePaths = {"/home/605/huynh/project/camera128.jpg", "/home/605/huynh/project/camera256.jpg", "/home/605/huynh/project/camera512.jpg"};

    if (rank==0){
        for(int i=0; i< static_cast<int>(imagePaths.size()); i++){
            cv::Mat image;
            if(!openImage(image, imagePaths[i])){
                std::cerr << "Failed to open the image at " << imagePaths[i] << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1); //ensure all processes not run avoiding further erroneous operations
                return 1;
            }
            //timer is reset for different image sizes, with auto start and auto end
            auto start = std::chrono::high_resolution_clock::now(); 

            std::vector<int> histogram = computeHistogram(image);
            std::vector<double> probability = computeProbability(histogram, image.total());
            int threshold = OtsuVarianceMPI(probability,rank,size);

            auto end = std::chrono::high_resolution_clock::now(); 

            std::chrono::duration<double> elapsed = end - start; 
            std::cout << "Parallel code using MPI takes " << elapsed.count() << "seconds to segment a " << image.rows << "x" << image.cols << "camera man image\n";
            cv::Mat segmented = getSegmentedImage(image,threshold);
            std::string outputPath = "/home/605/huynh/project/cameraOtsu" + std::to_string(image.rows) + "_MPI.jpg"; 
            cv::imwrite(outputPath, segmented);
            std::cout << "Segmented image save to " << outputPath << std::endl;
        }
    }
 
    MPI_Finalize();
    
    return 0;
} 
