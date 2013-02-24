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

void solve_one_channel_bis(const vector<Point>& z, const vector<Point>& x, const Mat& ref_image, Mat& image, int width){
  int nb_pixels = image.size().width*image.size().height;
  int image_height = image.size().height;
  vector<int> X(nb_pixels);
  vector<int> Z(nb_pixels);

  for(int k = 0; k < x.size(); ++k)
  {
    for(int i = x[k].x - width/2; i <= x[k].x + width/2; ++i)
    {
      for(int j = x[k].y - width/2; j <= x[k].y + width/2; ++j)
      {
        X[i*image_height+j] += 1;
        Z[i*image_height+j] += ref_image.at<unsigned char>(z[k].x +i-x[k].x,z[k].y+j-x[k].y);
      }
    }
  }

  for(int i = 0; i < image.size().width; ++i)
  {
    for(int j = 0; j < image.size().height; ++j)
    {
      int k = image_height*i+j;
      if(X[k] != 0){
        image.at<unsigned char>(i,j) = Z[k]/X[k];
      }
    }
  }
}

void fill_patch(real_2d_array& A, real_1d_array& b, int img_width, const Point& im_point, const Point& ref_point, int width, const Mat& ref_image)
{
  /*
  cout << "width " << width << endl;
  cout << "im_point.x " << im_point.x << endl;
  cout << "im_point.y " << im_point.y << endl;
  */

  // A is diagonal

  // loop through rows
  for(int icol = -width/2; icol <= width/2; ++icol)
  {
    for(int irow = -width/2; irow <= width/2; ++irow)
    {
      A[img_width*(im_point.y+icol)+im_point.x+irow][img_width*(im_point.y+icol)+im_point.x+irow] += 1;
      b[img_width*(im_point.y+icol)+im_point.x+irow] += ref_image.at<unsigned char>(ref_point.x+irow,ref_point.y+icol);
    }
  }
}

void solve_one_channel(const vector<Point>& z, const vector<Point>& x, const Mat& ref_image, Mat& image, int width)
{
  /*
  -----x----
  |
  y
  |
  */

  int nb_pixels = image.size().width*image.size().height;
  real_2d_array A;
  A.setlength(nb_pixels, nb_pixels);
  init_2d_array(A, nb_pixels, nb_pixels);

  real_1d_array b;
  b.setlength(nb_pixels);
  init_1d_array(b, nb_pixels);
  
  for(int i = 0; i < (int)x.size(); ++i)
  {
    fill_patch(A, b, image.size().width, x[i], z[i], width, ref_image); 
  }
 
  ae_int_t info = 1;
  densesolverlsreport rep;
  real_1d_array sol;
  sol.setlength(nb_pixels);
  
  rmatrixsolvels(A, nb_pixels, nb_pixels, b, 0.0, info, rep, sol);

  // Fill the image with solution(sol)
  for(int i = 0; i < image.size().height; ++i)
  {
    for(int j = 0; j < image.size().width; ++j)
    {
      image.at<unsigned char>(i,j) = sol(i*image.size().width+j);
    }
  }  
}

void solve_opt(const vector<Point>& z, const vector<Point>& x, const Mat& ref_image, Mat& image, int width)
{
  Mat im_one_channel(image.size(), CV_8UC1);
  Mat ref_one_channel(ref_image.size(), CV_8UC1);

  for(int ichannel = 0; ichannel < 3; ++ichannel)
  {
    for(int irow = 0; irow < image.size().height; ++irow)
      for(int icol = 0; icol < image.size().width; ++icol)
        im_one_channel.at<unsigned char>(irow, icol) = image.at<Vec3b>(irow, icol)[ichannel];

    for(int irow = 0; irow < ref_image.size().height; ++irow)
      for(int icol = 0; icol < ref_image.size().width; ++icol)
        ref_one_channel.at<unsigned char>(irow, icol) = ref_image.at<Vec3b>(irow, icol)[ichannel];

    solve_one_channel(z, x, ref_one_channel, im_one_channel, width);
    
    // Combine the image of each channel to form the final one
    for(int irow = 0; irow < image.size().height; ++irow)
      for(int icol = 0; icol < image.size().width; ++icol)
        image.at<Vec3b>(irow, icol)[ichannel] = im_one_channel.at<unsigned char>(irow, icol);  
  }
}

void solve_opt_bis(const vector<Point>& z, const vector<Point>& x, const Mat& ref_image, Mat& image, int width)
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

    solve_one_channel_bis(z, x, ref_one_channel, im_one_channel, width);
    
    // Combine the image of each channel to form the final one
    for(int irow = 0; irow < image.size().width; ++irow)
      for(int icol = 0; icol < image.size().height; ++icol)
        image.at<Vec3b>(irow, icol)[ichannel] = im_one_channel.at<unsigned char>(irow, icol);  
  }
}

void wsolve_one_channel_bis(const vector<Point>& z, const vector<Point>& x, const Mat& ref_image, Mat& image, int width, const vector<double>& weights){
  int nb_pixels = image.size().width*image.size().height;
  int image_height = image.size().height;
  vector<double> X(nb_pixels, 0);
  vector<double> Z(nb_pixels, 0);

  for(int k = 0; k < x.size(); ++k)
  {
    for(int i = x[k].x - width/2; i <= x[k].x + width/2; ++i)
    {
      for(int j = x[k].y - width/2; j <= x[k].y + width/2; ++j)
      {
        X[i*image_height+j] += weights[k];
        Z[i*image_height+j] += weights[k]*ref_image.at<unsigned char>(z[k].x +i-x[k].x,z[k].y+j-x[k].y);
      }
    }
  }

  for(int i = 0; i < image.size().width; ++i)
  {
    for(int j = 0; j < image.size().height; ++j)
    {
      int k = image_height*i+j;
      if(X[k] != 0){
        image.at<unsigned char>(i,j) = Z[k]/X[k];
      }
    }
  }
}

void wsolve_opt_bis(const vector<Point>& z, const vector<Point>& x, const Mat& ref_image, Mat& image, int width, const vector<double>& weights)
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

    wsolve_one_channel_bis(z, x, ref_one_channel, im_one_channel, width, weights);
    
    // Combine the image of each channel to form the final one
    for(int irow = 0; irow < image.size().width; ++irow)
      for(int icol = 0; icol < image.size().height; ++icol)
        image.at<Vec3b>(irow, icol)[ichannel] = im_one_channel.at<unsigned char>(irow, icol);  
  }
}

