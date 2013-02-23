#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <cmath>

#include "common.h"
#include "solve_opt.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
  Mat image;
  image = imread(argv[1], 1);

  if( argc != 2 || !image.data )
  {
    printf( "No image data \n" );
    return -1;
  }

  // Image synthesized
  Mat im(64, 64, CV_8UC3);

  // Generate grid points
  int w = 9;
  vector<Point> gridPoints;
  grid(gridPoints, w, image);

  // Generate random initialization
  vector<Point> randomPoints;
  randomNH(randomPoints, w, image, gridPoints); 

  solve_opt(randomPoints, gridPoints, image, im, w);

  namedWindow( "Display Image", CV_WINDOW_AUTOSIZE);
  imshow("Dispaly Image, syn_image", im);
  waitKey(0);
 
  return 0;
}
