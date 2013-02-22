#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <cmath>

using namespace cv;

double diff(const Vec3b& pa, const Vec3b& pb)
{
  double res = 0;
  for(int i = 0; i < 3; ++i)
    res += (pa[i]-pb[i])*(pa[i]-pb[i]);
  return sqrt(res);
}

int main(int argc, char** argv)
{
  Mat image;
  image = imread(argv[1], 1);

  if( argc != 2 || !image.data )
  {
    printf( "No image data \n" );
    return -1;
  }

  namedWindow( "Display Image", CV_WINDOW_AUTOSIZE);

  Mat_<Vec3b> img(64, 64);
  imshow("Disply Image, syn_image", img);
  waitKey();

  img(0,0) = Vec3b(0,255,0);
  return 0;

}
