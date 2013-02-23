#ifndef COMMON_H
#define COMMON_H
#endif

#include <cv.h>
#include <vector>

using namespace cv;
using namespace std;

// Difference of pixels
double diff(const Vec3b& pa, const Vec3b& pb);

// Difference of two neighborhoods
double diff_NH(const Point& pa, const Point& pb, const Mat& imagea, const Mat& imageb, const int& w);

// Compute the neighborhood of a pixel in a given image (neigborhood of size (w+1)*(w+1)
vector<Vec3b>  neighborhood(const Point& pixel, const int& w, const Mat& image);

// Find the nearest neighborhood in imageb (centered on the resulting Point) of the neighborhood defined by Point p and width w in imagea.
Point nearestNH(const Point& p, const int& w, const Mat& imagea, const Mat& imageb);

// Find the nearest neighborhoods in imageb of all the neighborhood in imagea of vector p.
void nearestNH(const vector<Point>& p, const int& w, const Mat& imagea, const Mat& imageb, vector<Point>& nNH);

// Put in gridPoints the points of a grid of width w/4 in image.
void grid(vector<Point>& gridPoints, const int& w, const Mat& image);

// Generate random neighborhoods for initialization.
void randomNH(vector<Point>& randomPoints, const int& w, const Mat& image, const vector<Point>& p);


