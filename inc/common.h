#ifndef COMMON_H
#define COMMON_H
#endif

#include <cv.h>
#include <vector>

// Difference of pixels
double diff(const cv::Vec3b& pa, const cv::Vec3b& pb);

// Compute the neighborhood of a pixel in a given image (neigborhood of size (w/2+1)*(w/2+1)
std::vector<cv::Vec3b>  neighborhood(const cv::Point& pixel, const int& w, const cv::Mat& image);




