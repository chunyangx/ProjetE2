#ifndef SOLVE_OPT_H
#define SOLVE_OPT_H
#endif

#include "common.h"
#include "cv.h"
#include <vector>

// R,G,B independent

/*
  For the solution of gradient matrix problem, we use alglib solvers. 
  We use alglib::sparsematrix to solve the linear problem, the later, by using conjugate gradient algorithm, this allows us to solve problem in O(n^2)
  N the number of pixels in the generating image. 
*/

// Solve energy optimization problem and render image, O(n), n the number of pixels in synthesized image
void solve_basic(const std::vector<cv::Point>& z, const std::vector<cv::Point>& x, const cv::Mat& ref_image, cv::Mat& render_image, int width);

// Solve energy optimization problem with gaussian weights, O(n), n the number of pixels in synthesized image
void solve_gaussian(const std::vector<cv::Point>& z, const std::vector<cv::Point>& x, const cv::Mat& ref_image, cv::Mat& render_image, int width);

// Solve energy optimization problem with gaussian weights and different patch weights, n the number of pixels in synthesized image 
void wsolve_gaussian(const std::vector<cv::Point>& z, const std::vector<cv::Point>& x, const cv::Mat& ref_image, cv::Mat& render_image, int width, const std::vector<double>& weights);

// Solve energy optimization problem with gradient in energy function, O(n^2), n the number of pixels in synthesized image 
void solve_grad(const std::vector<cv::Point>& z, const std::vector<cv::Point>& x, const cv::Mat& ref_image, cv::Mat& render_image, int width);

// Solve energy optimization problem with gradient in energy function and with gaussian weights, O(n^2), n the number of pixels in synthesized image 
void solve_ggrad(const std::vector<cv::Point>& z, const std::vector<cv::Point>& x, const cv::Mat& ref_image, cv::Mat& render_image, int width);

// Solve energy optimization problem with gradient in energy function and different patch weights, O(n^2), n the number of pixels in synthesized image 
void wsolve_grad(const std::vector<cv::Point>& z, const std::vector<cv::Point>& x, const cv::Mat& ref_image, cv::Mat& render_image, int width, const vector<double>& weights);

