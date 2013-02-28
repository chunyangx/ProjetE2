#ifndef SOLVE_OPT_H
#define SOLVE_OPT_H
#endif

#include "common.h"
#include "cv.h"
#include <vector>

// R,G,B independent

// Solve energy optimization problem and render image
void solve_basic(const std::vector<cv::Point>& z, const std::vector<cv::Point>& x, const cv::Mat& ref_image, cv::Mat& render_image, int width);

void solve_opt(const std::vector<cv::Point>& z, const std::vector<cv::Point>& x, const cv::Mat& ref_image, cv::Mat& render_image, int width);

void solve_opt_bis(const std::vector<cv::Point>& z, const std::vector<cv::Point>& x, const cv::Mat& ref_image, cv::Mat& render_image, int width);

void wsolve_opt_bis(const std::vector<cv::Point>& z, const std::vector<cv::Point>& x, const cv::Mat& ref_image, cv::Mat& render_image, int width, const std::vector<double>& weights);

void solve_opt_grad(const std::vector<cv::Point>& z, const std::vector<cv::Point>& x, const cv::Mat& ref_image, cv::Mat& render_image, int width, const std::vector<double>& weights);



