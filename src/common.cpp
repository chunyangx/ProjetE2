#include "common.h"
#include <cmath>
#include <vector>

using namespace cv;
using namespace std;

double diff(const Vec3b& pa, const Vec3b& pb)
{
  double res = 0;
  for(int i = 0; i < 3; ++i)
    res += (pa[i]-pb[i])*(pa[i]-pb[i]);
  return sqrt(res);
}

double diff_NH(const Point& pa, const Point& pb, const Mat& imagea, const Mat& imageb, const int& w)
{
  double res = 0;
  vector<Vec3b> NHa = neighborhood(pa,w,imagea);
  vector<Vec3b> NHb = neighborhood(pb,w,imageb);

  for(int i = 0; i < NHa.size(); ++i)
    res += diff(NHa[i],NHb[i]);
  return sqrt(res);
}

vector<Vec3b>  neighborhood(const Point& pixel, const int& w, const Mat& image)
{
  vector<Vec3b> neighborH;
  for(int x=pixel.x-w/4; x<=pixel.x+w/4; x++)
  {
    for(int y=pixel.y-w/4; y<=pixel.y+w/4; y++){
        neighborH.push_back(image.at<Vec3b>(x,y));
    }
  }
  return neighborH;
}
