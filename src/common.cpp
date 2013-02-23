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
  return res;
}

vector<Vec3b>  neighborhood(const Point& pixel, const int& w, const Mat& image)
{
  vector<Vec3b> neighborH;
  for(int x=pixel.x-w/2; x<=pixel.x+w/2; x++)
  {
    for(int y=pixel.y-w/2; y<=pixel.y+w/2; y++){
        neighborH.push_back(image.at<Vec3b>(x,y));
    }
  }
  return neighborH;
}

Point nearestNH(const Point& p, const int& w, const Mat& imagea, const Mat& imageb)
{
  double minDist = numeric_limits<double>::max();
  Point nNH(-1,-1);
  for(int x=w/2;x<imageb.rows-w/2;++x){
    for(int y=w/2;y<imageb.cols-w/2;++y){
        Point xy(x,y);
        double dist = diff_NH(p,xy,imagea,imageb,w);
        if (dist<minDist){
          minDist = dist;
          nNH = xy;
        }
    }
  }
  return nNH;
}

void nearestNH(const vector<Point>& p, const int& w, const Mat& imagea, const Mat& imageb, vector<Point>& nNH)
{
  for(int k=0;k<p.size();++k){
    nNH.push_back(nearestNH(p[k],w,imagea,imageb));
  }
}