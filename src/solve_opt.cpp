#include "solvers.h"
#include "common.h"
#include <stdexcept>
#include <iostream>

using namespace std;
using namespace alglib;
using namespace cv;

// For debug
void print_channel(const Mat& image, int ichannel)
{
  for(int irow = 0; irow < image.size().height; ++irow)
  {
    for(int icol = 0; icol < image.size().width; ++icol)
      cout << image.at<Vec3b>(irow, icol)[ichannel] << " ";
    cout << endl;
  }     
}

void print_matrix(const sparsematrix& A, int width)
{
  for(int i = 0; i <= width; ++i)
  {
    for(int j = 0; j <= width; ++j)
      cout << sparseget(A,i,j);
    cout << endl;
  }
}

void print_array(const real_1d_array& b, int length)
{
  for(int i = 0; i <= length; ++i)
    cout << b[i] << " ";
  cout << endl; 
}

// Initialization of array in alglib
void init_2d_array(real_2d_array& A, int nrow, int ncol)
{
  for(int i = 0; i < nrow; ++i)
    for(int j = 0; j < ncol; ++j)
      A[i][j] = 0;
}

void init_1d_array(real_1d_array& b, int length)
{
  for(int i = 0 ; i < length; ++i)
    b[i] = 0;
}

void solve_one_channel_gaussian(const vector<Point>& z, const vector<Point>& x, const Mat& ref_image, Mat& image, int width){
  int nb_pixels = image.rows*image.cols;
  int image_height = image.rows;

  vector<double> X(nb_pixels);
  vector<double> Z(nb_pixels);

  // gaussian width
  double gamma = width/2;

  // assign gaussian weights
  for(int k = 0; k < x.size(); ++k)
  {
    for(int irow = x[k].x - width/2; irow < x[k].x + width/2; ++irow)
    {
      for(int icol = x[k].y - width/2; icol < x[k].y + width/2; ++icol)
      {
        int x_NH = irow - x[k].x;
        int y_NH = icol - x[k].y;
        double gaussian_weight = exp(-(x_NH*x_NH+y_NH*y_NH)/(2*gamma*gamma));
        X[icol*image_height+irow] += gaussian_weight;
        Z[icol*image_height+irow] += gaussian_weight*ref_image.at<unsigned char>(z[k].x +irow-x[k].x,z[k].y+icol-x[k].y);
      }
    }
  }

  for(int irow = 0; irow < image.rows; ++irow)
  {
    for(int icol = 0; icol < image.cols; ++icol)
    {
      int k = image_height*icol+irow;
      if(X[k] != 0){
        image.at<unsigned char>(irow,icol) = Z[k]/X[k];
      }
    }
  }
}

/*
  Solve energy optimization problem and with gaussian weights render image, O(n), n the number of pixels in synthesized image.
  As the energy is quadratic in patch, E = |px-pz|^2
  For each patch, we apply a gaussian fall-off function with gaussian center the patch center, then, by giving the gradient expression equal to zero, we obtain:
    X[x,y] = (sum(gaussian(i)))*[gaussian(1)Z1+gaussian(2)Z2+...+gaussian(n)Zn], n the nearest neighbours of neighbour that contains pixels X[x,y], with gaussian_weight = exp(-(x_NH*x_NH+y_NH*y_NH)/(2*gamma*gamma)), x_NH, y_NH the distance to the center of the patch
  We could see that the gradient do not have crosses terms, so we calculate the above sum directly which gives O(n) complexity.  

  Parameters:
    x: all the centers that we consider(we don't consider each pixel to be a center)
    z: the corresponding nearest neighbours in the given texture of the points in x
    ref_image: the given texture image
    image: the image that we render
    width: the width of patch that we considier, generally 8, 16, 32  
*/

void solve_gaussian(const vector<Point>& z, const vector<Point>& x, const Mat& ref_image, Mat& image, int width)
{
  Mat im_one_channel(image.size(), CV_8UC1);
  Mat ref_one_channel(ref_image.size(), CV_8UC1);

  for(int ichannel = 0; ichannel < 3; ++ichannel)
  {
    for(int irow = 0; irow < image.rows; ++irow)
    {
      for(int icol = 0; icol < image.cols; ++icol)
      {
        im_one_channel.at<unsigned char>(irow, icol) = image.at<Vec3b>(irow, icol)[ichannel];
      }
    }

    for(int irow = 0; irow < ref_image.rows; ++irow)
    {
      for(int icol = 0; icol < ref_image.cols; ++icol)
      {
        ref_one_channel.at<unsigned char>(irow, icol) = ref_image.at<Vec3b>(irow, icol)[ichannel];
      }
    }

    solve_one_channel_gaussian(z, x, ref_one_channel, im_one_channel, width);
    
    // Combine the image of each channel to form the final one
    for(int irow = 0; irow < image.size().height; ++irow)
      for(int icol = 0; icol < image.size().width; ++icol)
        image.at<Vec3b>(irow, icol)[ichannel] = im_one_channel.at<unsigned char>(irow, icol);  
  }
}

void wsolve_one_channel_gaussian(const vector<Point>& z, const vector<Point>& x, const Mat& ref_image, Mat& image, int width, const vector<double>& weights){
  int nb_pixels = image.cols*image.rows;
  int image_height = image.rows;

  vector<double> X(nb_pixels, 0);
  vector<double> Z(nb_pixels, 0);
  double gamma = width/2;

  for(int k = 0; k < x.size(); ++k)
  {
    for(int irow = x[k].x - width/2; irow < x[k].x + width/2; ++irow)
    {
      for(int icol = x[k].y - width/2; icol < x[k].y + width/2; ++icol)
      {
        int x_NH = irow - x[k].x;
        int y_NH = icol - x[k].y;
        double gaussian_weight = exp(-(x_NH*x_NH+y_NH*y_NH)/(2*gamma*gamma));
        X[icol*image_height+irow] += weights[k]*gaussian_weight;
        Z[icol*image_height+irow] += weights[k]*gaussian_weight*ref_image.at<unsigned char>(z[k].x +irow-x[k].x,z[k].y+icol-x[k].y);
      }
    }
  }

  for(int irow = 0; irow < image.rows; ++irow)
  {
    for(int icol = 0; icol < image.cols; ++icol)
    {
      int k = image_height*icol+irow;
      if(X[k] != 0){
        image.at<unsigned char>(irow,icol) = Z[k]/X[k];
      }
    }
  }
}

/*
  Solve energy optimization problem and with gaussian weights and robust optimization and then render image, O(n), n the number of pixels in synthesized image.
  For each patch, we apply a gaussian fall-off function with gaussian center the patch center and here for a given patch, the function wsolve_gaussian implements an robust optimization process,  the energy is quadratic in patch becomes E = |px-pz|^0.8 instead of being quadratic. By giving the gradient expression equal to zero, we obtain:
    X[x,y] = (sum(gaussian(i)))*[gaussian(1)Z1+gaussian(2)Z2+...+gaussian(n)Zn], n the nearest neighbours of neighbour that contains pixels X[x,y], with gaussian_weight = exp(-(x_NH*x_NH+y_NH*y_NH)/(2*gamma*gamma)), x_NH, y_NH the distance to the center of the patch
  We could see that the gradient do not have crosses terms, so we calculate the above sum directly which gives O(n) complexity.

  Parameters:
    x: all the centers that we consider(we don't consider each pixel to be a center)
    z: the corresponding nearest neighbours in the given texture of the points in x
    ref_image: the given texture image
    image: the image that we render
    width: the width of patch that we considier, generally 8, 16, 32 
    weights: the weights attributed to each patch
*/

void wsolve_gaussian(const vector<Point>& z, const vector<Point>& x, const Mat& ref_image, Mat& image, int width, const vector<double>& weights)
{
  Mat im_one_channel(image.size(), CV_8UC1);
  Mat ref_one_channel(ref_image.size(), CV_8UC1);

  for(int ichannel = 0; ichannel < 3; ++ichannel)
  {
    for(int irow = 0; irow < image.size().height; ++irow)
    {
      for(int icol = 0; icol < image.size().width; ++icol)
      {
        im_one_channel.at<unsigned char>(irow, icol) = image.at<Vec3b>(irow, icol)[ichannel];
      }
    }

    for(int irow = 0; irow < ref_image.size().height; ++irow)
    {
      for(int icol = 0; icol < ref_image.size().width; ++icol)
      {
        ref_one_channel.at<unsigned char>(irow, icol) = ref_image.at<Vec3b>(irow, icol)[ichannel];
      }
    }

    wsolve_one_channel_gaussian(z, x, ref_one_channel, im_one_channel, width, weights);
    
    // Combine the image of each channel to form the final one
    for(int irow = 0; irow < image.size().height; ++irow)
      for(int icol = 0; icol < image.size().width; ++icol)
        image.at<Vec3b>(irow, icol)[ichannel] = im_one_channel.at<unsigned char>(irow, icol);  
  }
}

void solve_one_channel_grad(const vector<Point>& z, const vector<Point>& x, const Mat& ref_image, Mat& image, int width)
{
  /*
  -----x----
  |
  y
  |
  */

  int nb_pixels = image.size().width*image.size().height;
  int image_rows = image.rows;
 
  sparsematrix A;
  sparsecreate(nb_pixels, nb_pixels, A);

  real_1d_array b;
  b.setlength(nb_pixels);
  init_1d_array(b, nb_pixels);
  //double gamma = width/2;

  for(int k = 0; k < x.size(); ++k)
  {
    // Non gradient term
    for(int irow = x[k].x - width/2; irow < x[k].x + width/2; ++irow)
    {
      for(int icol = x[k].y - width/2; icol < x[k].y + width/2; ++icol)
      {
        /*int x_NH = i - x[k].x;
        int y_NH = j - x[k].y;
        double gaussian_weight = exp(-(x_NH*x_NH+y_NH*y_NH)/(2*gamma*gamma));*/
        double gaussian_weight = 1.;

        sparseadd(A, icol*image_rows+irow, icol*image_rows+irow, gaussian_weight);
        b[icol*image_rows+irow] += gaussian_weight*ref_image.at<unsigned char>(z[k].x + (irow-x[k].x), z[k].y + (icol-x[k].y));
      }
    }

    // Gradient term
    double mu = .1;
    for(int irow = x[k].x - width/2 +1; irow < x[k].x + width/2 -1; ++irow)
    {
      for(int icol = x[k].y - width/2 +1; icol < x[k].y + width/2 -1; ++icol)
      {
        /*int x_NH = i - x[k].x;
        int y_NH = j - x[k].y;
        double gaussian_weight = exp(-(x_NH*x_NH+y_NH*y_NH)/(2*gamma*gamma));*/
        double gaussian_weight = 1.;

        // X direction gradient
        sparseadd(A, icol*image_rows+irow+1, icol*image_rows+irow+1, mu*gaussian_weight);
        sparseadd(A, icol*image_rows+irow+1, icol*image_rows+irow-1, -mu*gaussian_weight);
        b[icol*image_rows+irow+1] += mu*gaussian_weight*(ref_image.at<unsigned char>(z[k].x + (irow-x[k].x)+1, z[k].y + (icol-x[k].y)) - ref_image.at<unsigned char>(z[k].x + (irow-x[k].x)-1, z[k].y + (icol-x[k].y)));
        
        sparseadd(A, icol*image_rows+irow-1, icol*image_rows+irow-1, mu*gaussian_weight);
        sparseadd(A, icol*image_rows+irow-1, icol*image_rows+irow+1, -mu*gaussian_weight);
        b[icol*image_rows+irow-1] += mu*gaussian_weight*(ref_image.at<unsigned char>(z[k].x + (irow-x[k].x)-1, z[k].y + (icol-x[k].y)) - ref_image.at<unsigned char>(z[k].x + (irow-x[k].x)+1, z[k].y + (icol-x[k].y)));
      
        // Y direction gradient
        sparseadd(A, icol*image_rows+irow+1, icol*image_rows+irow+1, mu*gaussian_weight);
        sparseadd(A, icol*image_rows+irow-1, icol*image_rows+irow+1, -mu*gaussian_weight);
        b[icol*image_rows+irow+1] += mu*gaussian_weight*(ref_image.at<unsigned char>(z[k].x + (irow-x[k].x), z[k].y + (icol-x[k].y)+1) - ref_image.at<unsigned char>(z[k].x + (irow-x[k].x), z[k].y + (icol-x[k].y)-1));

        sparseadd(A, icol*image_rows+irow-1, icol*image_rows+irow-1, mu*gaussian_weight);
        sparseadd(A, icol*image_rows+irow+1, icol*image_rows+irow-1, -mu*gaussian_weight);
        b[icol*image_rows+irow+1] += mu*gaussian_weight*(ref_image.at<unsigned char>(z[k].x + (irow-x[k].x), z[k].y + (icol-x[k].y)-1) - ref_image.at<unsigned char>(z[k].x + (irow-x[k].x), z[k].y + (icol-x[k].y)+1));
      }
    }
  }

  sparseconverttocrs(A);
  linlsqrstate s;
  linlsqrreport rep;
  real_1d_array sol;
  sol.setlength(nb_pixels);
  init_1d_array(sol, nb_pixels);

  // solve the problem
  linlsqrcreate(nb_pixels, nb_pixels, s);
  linlsqrsolvesparse(s, A, b);
  linlsqrresults(s, sol, rep);

  // Fill the image with solution(sol)
  for(int irow = 0; irow < image.rows; ++irow)
  {
    for(int icol = 0; icol < image.cols; ++icol)
    {
      int k = image_rows*icol+irow;
      image.at<unsigned char>(irow,icol) = sol[k];
    }
  }   
}

/*
  Solve energy optimization problem and with gaussian weights render image, O(n^2), n the number of pixels in synthesized image.
  The energy function in patch could be wriiten as: E = |px-pz|^2+mu*|Dx-Dz|^2
  We choose to express the gradient of pixel (xx,yy) as Dxx = pixel(x+1)-pixel(x-1) we could not perform as before as we have crossed gradient term, but the problem of gradient equals to zeros remains linear, so we solve linear problem Ax = b.  
  When calculating the gradient matrix considering only the derivate in x direction, the contribution of one pixel in the gradient matrix A and demanded value b is(we choose mu = 0.1):
  derivate / [x*heignt+y, x*height+y]
  in A for [x*heignt+y, x*height+y]:2              b [x*height+y]: z(x,y)

  derivate / [x*heignt+y+1, x*height+y+1]
  in A for [x*height+y+1, x*height+y+1] : mu*2
     A for [x*height+y+1, x*height+y-1] : mu*-2       b [x*height+y+1]: z(x+1,y)-z(x-1,y) 
     A for [x*height+y-1, x*height+y-1] : mu*2
     A for [x*height+y+1, x*height+y-1] : mu*-2       b [x*height+y-1]: z(x-1,y)-z(x+1,y)

  The similar is for derivate / [x*height+y-1, x*]

  The matrix A becomes 2 0 -2mu 
                         2 0 -2mu
                           2 0 -2mu etc.

  Parameters:
    x: all the centers that we consider(we don't consider each pixel to be a center)
    z: the corresponding nearest neighbours in the given texture of the points in x
    ref_image: the given texture image
    image: the image that we render
    width: the width of patch that we considier, generally 8, 16, 32 
*/

void solve_grad(const std::vector<cv::Point>& z, const std::vector<cv::Point>& x, const cv::Mat& ref_image, cv::Mat& image, int width)
{
  Mat im_one_channel(image.size(), CV_8UC1);
  Mat ref_one_channel(ref_image.size(), CV_8UC1);

  for(int ichannel = 0; ichannel < 3; ++ichannel)
  {
    for(int irow = 0; irow < image.rows; ++irow)
    {
      for(int icol = 0; icol < image.cols; ++icol)
      {
        im_one_channel.at<unsigned char>(irow, icol) = image.at<Vec3b>(irow, icol)[ichannel];
      }
    }

    for(int irow = 0; irow < ref_image.rows; ++irow)
    {
      for(int icol = 0; icol < ref_image.cols; ++icol)
      {
        ref_one_channel.at<unsigned char>(irow, icol) = ref_image.at<Vec3b>(irow, icol)[ichannel];
      }
    }

    solve_one_channel_grad(z, x, ref_one_channel, im_one_channel, width);
    
    // Combine the image of each channel to form the final one
    for(int irow = 0; irow < image.rows; ++irow)
      for(int icol = 0; icol < image.cols; ++icol)
        image.at<Vec3b>(irow, icol)[ichannel] = im_one_channel.at<unsigned char>(irow, icol);  
  }
}

void solve_one_channel_basic(const vector<Point>& z, const vector<Point>& x, const Mat& ref_image, Mat& image, int width){
  int nb_pixels = image.size().width*image.size().height;
  int image_rows = image.rows;

  vector<double> X(nb_pixels, 0);
  vector<double> Z(nb_pixels, 0);

  for(int k = 0; k < x.size(); ++k)
  {
    // For every neighbour, modify all pixel coeffiecents         
    for(int irow = -width/2 + x[k].x; irow < width/2 + x[k].x; ++irow)
    {
      for(int icol = -width/2 + x[k].y; icol < width/2 + x[k].y; ++icol)
      { 
        X[irow + icol*image_rows] += 1;
        Z[irow + icol*image_rows] += ref_image.at<unsigned char>(z[k].x+(irow - x[k].x), z[k].y+(icol - x[k].y));
      }
    }
  }

  for(int irow = 0; irow < image.rows; ++irow)
  {
    for(int icol = 0; icol < image.cols; ++icol)
    {
      int k = image_rows*icol+irow;
      if (X[k] != 0)
        image.at<unsigned char>(irow,icol) = Z[k]/X[k];
    }
  }
}

/*
  Solve energy optimization problem and render image, O(n), n the number of pixels in synthesized image.
  As the energy is quadratic in patch, E = |px-pz|^2
  By giving the gradient expression equal to zero, we obtain:
    X[x,y] = (1/n)*[Z1+Z2+...+Zn], n the nearest neighbours of neighbour that contains pixels X[x,y]
  We could see that the gradient do not have crosses terms, so we calculate the above sum directly which gives O(n) complexity.

  Parameters:
    x: all the centers that we consider(we don't consider each pixel to be a center)
    z: the corresponding nearest neighbours in the given texture of the points in x
    ref_image: the given texture image
    image: the image that we render
    width: the width of patch that we considier, generally 8, 16, 32   
*/

void solve_basic(const std::vector<cv::Point>& z, const std::vector<cv::Point>& x, const cv::Mat& ref_image, cv::Mat& image, int width)
{
  Mat im_one_channel(image.size(), CV_8UC1);
  Mat ref_one_channel(ref_image.size(), CV_8UC1);

  // Solve the problem channel by channel
  for(int ichannel = 0; ichannel < 3; ++ichannel)
  {
    for(int irow = 0; irow < image.rows; ++irow)
    {
      for(int icol = 0; icol < image.cols; ++icol)
      {
        im_one_channel.at<unsigned char>(irow, icol) = image.at<Vec3b>(irow, icol)[ichannel];
      }
    }

    for(int irow = 0; irow < ref_image.rows; ++irow)
    {
      for(int icol = 0; icol < ref_image.cols; ++icol)
      {
        ref_one_channel.at<unsigned char>(irow, icol) = ref_image.at<Vec3b>(irow, icol)[ichannel];
      }
    }

    solve_one_channel_basic(z, x, ref_one_channel, im_one_channel, width);
    
    // Combine the image of each channel to form the final one
    for(int irow = 0; irow < image.rows; ++irow)
      for(int icol = 0; icol < image.cols; ++icol)
        image.at<Vec3b>(irow, icol)[ichannel] = im_one_channel.at<unsigned char>(irow, icol);  
  }
}

void solve_one_channel_ggrad(const vector<Point>& z, const vector<Point>& x, const Mat& ref_image, Mat& image, int width)
{
  /*
  -----x----
  |
  y
  |
  */

  int nb_pixels = image.size().width*image.size().height;
  int image_rows = image.rows;
 
  sparsematrix A;
  sparsecreate(nb_pixels, nb_pixels, A);

  real_1d_array b;
  b.setlength(nb_pixels);
  init_1d_array(b, nb_pixels);
  double gamma = width/2;

  for(int k = 0; k < x.size(); ++k)
  {
    // Non gradient term
    for(int irow = x[k].x - width/2; irow < x[k].x + width/2; ++irow)
    {
      for(int icol = x[k].y - width/2; icol < x[k].y + width/2; ++icol)
      {
        int x_NH = irow - x[k].x;
        int y_NH = icol - x[k].y;
        double gaussian_weight = exp(-(x_NH*x_NH+y_NH*y_NH)/(2*gamma*gamma));

        sparseadd(A, icol*image_rows+irow, icol*image_rows+irow, gaussian_weight);
        b[icol*image_rows+irow] += gaussian_weight*ref_image.at<unsigned char>(z[k].x + (irow-x[k].x), z[k].y + (icol-x[k].y));
      }
    }

    // Gradient term
    double mu = .1;
    for(int irow = x[k].x - width/2 +1; irow < x[k].x + width/2 -1; ++irow)
    {
      for(int icol = x[k].y - width/2 +1; icol < x[k].y + width/2 -1; ++icol)
      {
        int x_NH = irow - x[k].x;
        int y_NH = icol - x[k].y;
        double gaussian_weight = exp(-(x_NH*x_NH+y_NH*y_NH)/(2*gamma*gamma));

        // X direction gradient
        sparseadd(A, icol*image_rows+irow+1, icol*image_rows+irow+1, mu*gaussian_weight);
        sparseadd(A, icol*image_rows+irow+1, icol*image_rows+irow-1, -mu*gaussian_weight);
        b[icol*image_rows+irow+1] += mu*gaussian_weight*(ref_image.at<unsigned char>(z[k].x + (irow-x[k].x)+1, z[k].y + (icol-x[k].y)) - ref_image.at<unsigned char>(z[k].x + (irow-x[k].x)-1, z[k].y + (icol-x[k].y)));
        
        sparseadd(A, icol*image_rows+irow-1, icol*image_rows+irow-1, mu*gaussian_weight);
        sparseadd(A, icol*image_rows+irow-1, icol*image_rows+irow+1, -mu*gaussian_weight);
        b[icol*image_rows+irow-1] += mu*gaussian_weight*(ref_image.at<unsigned char>(z[k].x + (irow-x[k].x)-1, z[k].y + (icol-x[k].y)) - ref_image.at<unsigned char>(z[k].x + (irow-x[k].x)+1, z[k].y + (icol-x[k].y)));
      
        // Y direction gradient
        sparseadd(A, icol*image_rows+irow+1, icol*image_rows+irow+1, mu*gaussian_weight);
        sparseadd(A, icol*image_rows+irow-1, icol*image_rows+irow+1, -mu*gaussian_weight);
        b[icol*image_rows+irow+1] += mu*gaussian_weight*(ref_image.at<unsigned char>(z[k].x + (irow-x[k].x), z[k].y + (icol-x[k].y)+1) - ref_image.at<unsigned char>(z[k].x + (irow-x[k].x), z[k].y + (icol-x[k].y)-1));

        sparseadd(A, icol*image_rows+irow-1, icol*image_rows+irow-1, mu*gaussian_weight);
        sparseadd(A, icol*image_rows+irow+1, icol*image_rows+irow-1, -mu*gaussian_weight);
        b[icol*image_rows+irow+1] += mu*gaussian_weight*(ref_image.at<unsigned char>(z[k].x + (irow-x[k].x), z[k].y + (icol-x[k].y)-1) - ref_image.at<unsigned char>(z[k].x + (irow-x[k].x), z[k].y + (icol-x[k].y)+1));
      }
    }
  }

  sparseconverttocrs(A);
  linlsqrstate s;
  linlsqrreport rep;
  real_1d_array sol;
  sol.setlength(nb_pixels);
  init_1d_array(sol, nb_pixels);

  // solve the problem
  linlsqrcreate(nb_pixels, nb_pixels, s);
  linlsqrsolvesparse(s, A, b);
  linlsqrresults(s, sol, rep);

  // Fill the image with solution(sol)
  for(int irow = 0; irow < image.rows; ++irow)
  {
    for(int icol = 0; icol < image.cols; ++icol)
    {
      int k = image_rows*icol+irow;
      image.at<unsigned char>(irow,icol) = sol[k];
    }
  }   
}

/*
  The solve_ggrad function has the same structure as solve_grad.
  It adds gaussian fall-off function above solve_grad.
  For reference, see function: solve_grad, solve_gaussian

  Parameters:
    x: all the centers that we consider(we don't consider each pixel to be a center)
    z: the corresponding nearest neighbours in the given texture of the points in x
    ref_image: the given texture image
    image: the image that we render
    width: the width of patch that we considier, generally 8, 16, 32  
*/

void solve_ggrad(const std::vector<cv::Point>& z, const std::vector<cv::Point>& x, const cv::Mat& ref_image, cv::Mat& image, int width)
{
  Mat im_one_channel(image.size(), CV_8UC1);
  Mat ref_one_channel(ref_image.size(), CV_8UC1);

  for(int ichannel = 0; ichannel < 3; ++ichannel)
  {
    for(int irow = 0; irow < image.rows; ++irow)
    {
      for(int icol = 0; icol < image.cols; ++icol)
      {
        im_one_channel.at<unsigned char>(irow, icol) = image.at<Vec3b>(irow, icol)[ichannel];
      }
    }

    for(int irow = 0; irow < ref_image.rows; ++irow)
    {
      for(int icol = 0; icol < ref_image.cols; ++icol)
      {
        ref_one_channel.at<unsigned char>(irow, icol) = ref_image.at<Vec3b>(irow, icol)[ichannel];
      }
    }

    solve_one_channel_ggrad(z, x, ref_one_channel, im_one_channel, width);
    
    // Combine the image of each channel to form the final one
    for(int irow = 0; irow < image.rows; ++irow)
      for(int icol = 0; icol < image.cols; ++icol)
        image.at<Vec3b>(irow, icol)[ichannel] = im_one_channel.at<unsigned char>(irow, icol);  
  }
}
