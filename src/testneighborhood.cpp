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

  if( argc != 2 || !image.data )
  {
    printf( "No image data \n" );
    return -1;
  }
  

  PointTexture pixel(100,100);

  printf("%f\n",diff(image.at<Vec3b>(80,110),image.at<Vec3b>(120,100)));

  vector<Vec3b> neighborH = neighborhood(pixel,8,image);
  printf("%i\n",(int) neighborH.size());
  for(int i=0;i<neighborH.size();i++){
    for(int k=0;k<3;k++){
      printf("%i\n",i);
      printf("%u\n",neighborH[i][k]);
    }
  }
}
