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


void main_loop(const Mat& texture_ref,Mat& texture,const int w, int& random_init)
{
  // Compute all the neighborhood of picture image
  vector<Point> NHz;
  allPoints(NHz, w, texture_ref);
  Node* root = constructTree(NHz, w, texture_ref);

  // Generate grid points
  vector<Point> x;
  grid(x, w, texture);

  vector<Point> z;
  vector<double> weights(x.size(), 1);
  if(random_init){
    // Generate random initialization
    randomNH(z, w, texture_ref, x);
    random_init = 0;
  }else{
    findTreeNNH(x, w, texture, texture_ref, root, z);
    update_weights(x, z, w, texture, texture_ref, weights);
  }

  vector<Point> z_old;

  int k=0;
  int max_step = 100;
  while(z!=z_old && k<max_step){
 
    printf("%d\n",k);
    z_old = z;

    //solve_opt_bis(z, x, texture_ref, texture, w);
    wsolve_opt_bis(z, x, texture_ref, texture, w, weights);
    findTreeNNH(x, w, texture, texture_ref, root, z);
    update_weights(x, z, w, texture, texture_ref, weights);
    k++;
  }
}


int main(int argc, char** argv)
{
  Mat texture_ref;
  texture_ref = imread(argv[1], 1);

  if( argc != 2 || !texture_ref.data )
  {
    printf( "No image data \n" );
    return -1;
  }

  //texture synthesized (at scale 1/8 in order to start the optimization)
  Mat texture(32, 32, CV_8UC3);

  Mat texture_ref_resize;
  float scale = 1./4.; //Current resolution on which we are working
  int random_init = 1; //1 if we have to init randomly z for the first step
  int w_scale = 8; //Size of w we start using at the current resolution (8 for 1/4, 16 for 1/2, 32 for 1)
  
   for(int k=0;k<3;k++) //Loop over the different resolutions.
   {
    printf("resolution : %f --------\n",scale);
    resize(texture_ref, texture_ref_resize, Size(), scale, scale, INTER_LINEAR);
    resize(texture, texture, Size(), 2., 2., INTER_LINEAR);

    namedWindow( "Display Image", CV_WINDOW_AUTOSIZE);
    imshow("Dispaly Image, syn_image", texture_ref_resize);
    printf("Current texture of reference\n",scale);
    waitKey(0);
    namedWindow( "Display Image", CV_WINDOW_AUTOSIZE);
    imshow("Dispaly Image, syn_image", texture);
    printf("Current texture\n",scale);
    waitKey(0);

    scale = scale*2.;
    w_scale = 2*w_scale;
    int w = w_scale;

    for(int l=0;l<k+1;l++){
      w = w/2;
      printf("w : %d --------\n",w);
      main_loop(texture_ref_resize, texture, w, random_init);

      namedWindow( "Display Image", CV_WINDOW_AUTOSIZE);
      imshow("Dispaly Image, syn_image", texture);
      printf("Current texture\n",scale);
      waitKey(0);
    }
  }


  return 0;
}
