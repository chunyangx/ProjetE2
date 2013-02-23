#ifndef COMMON_H
#define COMMON_H
#endif

#include <cv.h>
#include <vector>

/*
class EPoint
{
public:
  EPoint(int x, int y): _x(x), _y(y) {}
  int x(){return _x;}
  int y(){return _y;}
>>>>>>> fix call
private:
  // Coordinates
  int _x;
  int _y;
};
*/

// Difference of pixels
double diff(const cv::Vec3b& pa, const cv::Vec3b& pb);

// Compute the neighborhood of a pixel in a given image (neigborhood of size (w/2+1)*(w/2+1)
std::vector<cv::Vec3b>  neighborhood(const PointTexture pixel, const int w, const cv::Mat image);




