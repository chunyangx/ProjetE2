#ifndef SOLVE_OPT_H
#define SOLVE_OPT_H
#endif

#include "common.h"
#include "cv.h"
#include <vector>

// R,G,B independent

// Solve energy optimization problem and render image
void solve_opt(const std::vector<Point>& z, const Mat& ref_image, Mat& render_image, int width);


