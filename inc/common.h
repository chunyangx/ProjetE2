#ifndef COMMON_H
#define COMMON_H
#endif

#include <cv.h>

class Point
{
public:
  Point(int x, int y): _x(x), _y(y) {}
  int x(){return _x;}
  int y(){return _y;}
private:
  // Coordinates
  int _x;
  int _y;
};

// Difference of pixels
double diff(const cv::Vec3b& pa, const cv::Vec3b& pb);




