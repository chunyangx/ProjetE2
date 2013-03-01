#ifndef SOLVE_OPT_H
#define SOLVE_OPT_H
#endif

#include "common.h"
#include "cv.h"
#include <vector>

// R,G,B independent

// Solve energy optimization problem and render image
void solve_basic(const std::vector<cv::Point>& z, const std::vector<cv::Point>& x, const cv::Mat& ref_image, cv::Mat& render_image, int width);

// Solve energy optimization problem with gaussian weights
void solve_gaussian(const std::vector<cv::Point>& z, const std::vector<cv::Point>& x, const cv::Mat& ref_image, cv::Mat& render_image, int width);

// Solve energy optimization problem with gaussian weights and different patch weights 
void wsolve_gaussian(const std::vector<cv::Point>& z, const std::vector<cv::Point>& x, const cv::Mat& ref_image, cv::Mat& render_image, int width, const std::vector<double>& weights);

// Solve energy optimization problem with gradient in energy function 
void solve_grad(const std::vector<cv::Point>& z, const std::vector<cv::Point>& x, const cv::Mat& ref_image, cv::Mat& render_image, int width);

// Solve energy optimization problem with gradient in energy function and with gaussia weights
void solve_ggrad(const std::vector<cv::Point>& z, const std::vector<cv::Point>& x, const cv::Mat& ref_image, cv::Mat& render_image, int width);


