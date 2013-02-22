#include <cv.h>

class Point
{
public:
  x(){return _x;}
  y(){return _y;}
private:
  // Coordinates
  int _x;
  int _y;
};

// Difference of pixels
double diff(const cv::Vec3b& pa, const cv::Vec3b& pb);


