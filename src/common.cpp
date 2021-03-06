#include "common.h"
#include <cmath>
#include <vector>

#include <stdio.h>      // For NULL
#include <stdlib.h>     // For random numbers    
#include <time.h>       // For random seeds

using namespace cv;
using namespace std;

// Difference of pixels
int diff(const Vec3b& pa, const Vec3b& pb)
{
  int res = 0;
  for(int i = 0; i < 3; ++i)
    res += (pa[i]-pb[i])*(pa[i]-pb[i]);
  return res;
}

// Difference of two neighborhoods
double diff_NH(const Point& pa, const Point& pb, const Mat& imagea, const Mat& imageb, const int& w)
{
  double res = 0.;
  vector<Vec3b> NHa = neighborhood(pa,w,imagea);
  vector<Vec3b> NHb = neighborhood(pb,w,imageb);

  int i = 0;
  for(int x=-w/2; x<w/2; x++)
  {
    for(int y=-w/2; y<w/2; y++){
      res += diff(NHa[i],NHb[i]);
      i++;
    }
  }
  return res;
}

/*double diff_NH(const Point& pa, const Point& pb, const Mat& imagea, const Mat& imageb, const int& w)
{
  double gamma = w/2;
  double res = 0.;
  vector<Vec3b> NHa = neighborhood(pa,w,imagea);
  vector<Vec3b> NHb = neighborhood(pb,w,imageb);

  int i = 0;
  for(int x=-w/2; x<w/2; x++)
  {
    for(int y=-w/2; y<w/2; y++){
      res += exp(-(x*x+y*y)/(2*gamma*gamma))*diff(NHa[i],NHb[i]);

      i++;
    }
  }
  return res;
}*/

// Compute the neighborhood of a pixel in a given image (neigborhood of size (w*w)
vector<Vec3b>  neighborhood(const Point& pixel, const int& w, const Mat& image)
{
  vector<Vec3b> neighborH;
  for(int x=pixel.x-w/2; x<pixel.x+w/2; x++)
  {
    for(int y=pixel.y-w/2; y<pixel.y+w/2; y++){
        neighborH.push_back(image.at<Vec3b>(x,y));
    }
  }
  return neighborH;
}

// Find the nearest neighborhood of a given one in a set of neighborhoods.
Point nearestNH(const Point& p, const int& w, const Mat& imagea, const vector<Point>& pb, const Mat& imageb)
{
  double minDist = numeric_limits<double>::max();
  Point nNH(-1,-1);
  for(int i=0;i<pb.size();++i){
    double dist = diff_NH(p,pb[i],imagea,imageb,w);
    if (dist<minDist){
      minDist = dist;
      nNH = pb[i];
    }
  }
  return nNH;
}

// Find the nearest neighborhood in imageb (centered on the resulting Point) of the neighborhood defined by Point p and width w in imagea.
Point nearestNH(const Point& p, const int& w, const Mat& imagea, const Mat& imageb)
{
  double minDist = numeric_limits<double>::max();
  Point nNH(-1,-1);
  for(int x=w/2;x<=imageb.rows-w/2;++x){
    for(int y=w/2;y<=imageb.cols-w/2;++y){
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

// Find the nearest neighborhoods in imageb of all the neighborhood in imagea of vector p.
void nearestNH(const vector<Point>& p, const int& w, const Mat& imagea, const Mat& imageb, vector<Point>& nNH)
{
  nNH.clear();
  for(int k=0;k<p.size();++k){
    nNH.push_back(nearestNH(p[k],w,imagea,imageb));
  }
}

// Put in gridPoints the points of a grid of width w/4 in image.
void grid(vector<Point>& gridPoints, const int& w, const Mat& image)
{
  gridPoints.clear();
  for(int x=w/2;x<=image.rows-w/2;x+=w/4){
    for(int y=w/2;y<=image.cols-w/2;y+=w/4){
        Point xy(x,y);
        gridPoints.push_back(xy);
    }
  }
}

// Returns the set of all possible neighborhood in the given image.
void allPoints(vector<Point>& gridPoints, const int& w, const Mat& image)
{
  gridPoints.clear();
  for(int x=w/2;x<=image.rows-w/2;x++){
    for(int y=w/2;y<=image.cols-w/2;y++){
        Point xy(x,y);
        gridPoints.push_back(xy);
    }
  }
}

// Generate random neighborhoods for initialization.
void randomNH(vector<Point>& randomPoints, const int& w, const Mat& image, const vector<Point>& p)
{
  randomPoints.clear();
  srand (time(NULL));
  int min_x = w/2;
  int min_y = w/2;
  int max_x = image.rows-w/2;
  int max_y = image.cols-w/2;

  for(int k=0;k<p.size();++k){
    Point xy(rand()%(max_x-min_x)+min_x, rand()%(max_y-min_y)+min_y);
    randomPoints.push_back(xy);
  }
}

// Calculate weights on each grid point
void update_weights(const vector<Point>& x, const vector<Point>& z, int width, const Mat& im, const Mat& image, vector<double>& weights)
{
  for(int i = 0; i < weights.size(); ++i)
  {
    double difference = diff_NH(x[i], z[i], im, image, width);
    double r = .8;
    if (difference != 0){
      weights[i] = pow(difference, (r-2.)/2.);
    }else{
      weights[i] = pow(exp(-1), (r-2.)/2.);
    }
  }   
}
