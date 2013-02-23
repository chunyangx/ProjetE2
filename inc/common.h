#ifndef COMMON_H
#define COMMON_H
#endif

#include <cv.h>
#include <vector>

// Difference of pixels
double diff(const cv::Vec3b& pa, const cv::Vec3b& pb);

// Difference of two neighborhoods
double diff_NH(const cv::Point& pa, const cv::Point& pb, const cv::Mat& imagea, const cv::Mat& imageb, const int& w);

// Compute the neighborhood of a pixel in a given image (neigborhood of size (w+1)*(w+1)
std::vector<cv::Vec3b>  neighborhood(const cv::Point& pixel, const int& w, const cv::Mat& image);

cv::Point nearestNH(const cv::Point& p, const int& w, const cv::Mat& imagea, const cv::Mat& imageb);

