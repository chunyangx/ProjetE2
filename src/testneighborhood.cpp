#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <cmath>
#include "common.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
  Mat image;
  image = imread(argv[1], 1);

  Mat imagebis;
  imagebis = imread(argv[2], 1);

  if( argc != 3 || !image.data )
  {
    printf( "No image data \n" );
    return -1;
  }
  

  Point pixel(100,100);
  Point nn = nearestNH(pixel, 5, image, imagebis);
  printf("Nearest neighborhood of coordinates : %i %i\n",nn.x,nn.y);
  Point pixel2(100,150);
  Point nn2 = nearestNH(pixel2, 5, image, imagebis);
  printf("Nearest neighborhood of coordinates : %i %i\n",nn2.x,nn2.y);
  vector<Point> pixel_vect;
  pixel_vect.push_back(pixel);
  pixel_vect.push_back(pixel2);
  vector<Point> nn_vect;
  nearestNH(pixel_vect, 5, image, imagebis,nn_vect);
  printf("Nearest neighborhood of coordinates : %i %i\n",nn_vect[0].x,nn_vect[0].y);
  printf("Nearest neighborhood of coordinates : %i %i\n",nn_vect[1].x,nn_vect[1].y);

  vector<Point> pixel_grid;
  grid(pixel_grid, 100, image);
  nearestNH(pixel_grid, 5, image, imagebis,nn_vect);
  printf("done\n");


}
