#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <cmath>

#include "common.h"
#include "kmeans.h"
#include "solve_opt.h"

using namespace cv;
using namespace std;

template<typename T>
void print_vec(const vector<T>& vec)
{
  for(int i = 0; i < vec.size(); ++i)
    cout << vec[i] << endl;
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

  //Image synthesized
  Mat im(128, 128, CV_8UC3);

  /*Mat im;
  im = imread(argv[2], 1);*/

  int w = 16;

  vector<Point> NHz;
  allPoints(NHz, w, image);
  Node* root = constructTree(NHz, w, image);

  // Generate grid points
  vector<Point> x;
  grid(x, w, im);

  // Generate random initialization
  vector<Point> z;
  randomNH(z, w, image, x); 

  vector<Point> z_old;

  vector<double> weights(x.size(), 1);

  int k=0;
  while(z!=z_old){
  printf("%d\n",k);
  z_old = z;
  //solve_opt_bis(z, x, image, im, w);
  wsolve_opt_bis(z, x, image, im, w, weights);

  findTreeNNH(x, w, im, image, root, z);
  update_weights(x, z, w, im, image, weights);
  k++;
  namedWindow( "Display Image", CV_WINDOW_AUTOSIZE);
  imshow("Dispaly Image, syn_image", im);
  waitKey(0);
  }

  return 0;
}
