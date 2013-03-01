#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <cmath>
#include <string>

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

// Main loop in the optimization process of the synthesis of a new texture
void main_loop(const Mat& texture_ref,Mat& texture,const int w, int& random_init)
{
  // Compute all the neighborhood of the given texture (texture_ref)
  vector<Point> NHz;
  allPoints(NHz, w, texture_ref);
  Node* root = constructTree(NHz, w, texture_ref);

  // Generate grid points x
  vector<Point> x;
  grid(x, w, texture);

  // Points z related to x
  vector<Point> z;

  // Weights for the IRLS
  vector<double> weights(x.size(), 1);

  //randil_init == 1 only at the beginning of the whole process
  if(random_init){
    // Generate random initialization of z
    randomNH(z, w, texture_ref, x);
    random_init = 0;
  }else{
    // Find the best possible neighborhoods z and update the weights
    findTreeNNH(x, w, texture, texture_ref, root, z);
    update_weights(x, z, w, texture, texture_ref, weights);
  }


  // Needed for the stop criterion.
  vector<Point> z_old;

  int k=0; //Number of iterations
  int max_step = 100;
  while(z!=z_old && k<max_step){ //Continue until the neighborhoods in the reference texture don't change or we reach the max_step limit
 
    printf("Iteration : %d\n",k); // Print the iteration
    z_old = z;

    //solve_opt_bis(z, x, texture_ref, texture, w);
    //wsolve_opt_bis(z, x, texture_ref, texture, w, weights);
    //solve_basic(z, x, texture_ref, texture, w);
    solve_opt_grad(z, x, texture_ref, texture, w, weights);
    findTreeNNH(x, w, texture, texture_ref, root, z);
    update_weights(x, z, w, texture, texture_ref, weights);

    k++;
  }
}


int main(int argc, char** argv)
{
  // Check that the number of arguments is correct
  if( argc < 3)
  {
    printf("Wrong arguments.\n 'Texture_Synthesis image.ext repertory/' generates a new texture (256x256) for image.ext and save the generated textures in repertory/\n");
    return -1;
  }

  // Load the reference texture image
  Mat texture_ref;
  texture_ref = imread(argv[1], 1);

  // Path used for saving the generated textures
  string path(argv[2]);

  //texture synthesized (at scale 1/8 in order to start the optimization)
  Mat texture(16, 16, CV_8UC3);

  Mat texture_ref_resize; // Will be the resized version of the reference texture
  float scale = 1./4.; //Current resolution on which we are working
  int random_init = 1; //1 if we have to init randomly z for the first step
  int w_scale = 8; //Size of w we start using at the current resolution (8 for 1/4, 16 for 1/2, 32 for 1)
  
   for(int k=0;k<3;k++) //Loop over the different resolutions.
   {
    printf("\nResolution : %f\n\n",scale);

    // Resize the textures
    resize(texture_ref, texture_ref_resize, Size(), scale, scale, INTER_LINEAR); // Resize reference texture in texture_ref_resize
    resize(texture, texture, Size(), 2., 2., INTER_LINEAR); // Resize the texture we are generating.

    // Modify the scale and init w according to the current resolution
    scale = scale*2.;
    w_scale = 2*w_scale;
    int w = w_scale;

    stringstream convertres; //used in order to save the generated textures
    convertres << k+1;

    for(int l=0;l<k+1;l++){ // Loop over the w (width of the neighborhoods)
      w = w/2; // new w
      printf("New w : %d\n\n",w);

      // Optimize the new texture for the current resolution and w
      main_loop(texture_ref_resize, texture, w, random_init);

      stringstream convertw; //used in order to save the generated textures
      convertw << w;

      // Save the current generated texture
      imwrite( path+"_res"+convertres.str()+"_w"+convertw.str()+".jpg", texture);
    }
  }
  printf("done\n");
  return 0;
}
