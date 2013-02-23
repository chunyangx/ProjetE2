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
  printf("%f\n",diff_NH(pixel, pixel, image, imagebis, 5));

  /*vector<Vec3b> neighborH = neighborhood(pixel,8,image);
  printf("%i\n",(int) neighborH.size());
  for(int i=0;i<neighborH.size();i++){
    for(int k=0;k<3;k++){
      printf("%i\n",i);
      printf("%u\n",neighborH[i][k]);
    }
  }*/
}
