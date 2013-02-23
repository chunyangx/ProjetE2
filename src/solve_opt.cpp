#include "solvers.h"
#include "common.h"
#include <stdexcept>
#include <iostream>

using namespace std;
using namespace alglib;
using namespace cv;

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

void fill_patch(real_2d_array& A, real_1d_array& b, int img_width, const Point& im_point, const Point& ref_point, int width, const Mat& ref_image)
{
  /*
  cout << "width " << width << endl;
  cout << "im_point.x " << im_point.x << endl;
  cout << "im_point.y " << im_point.y << endl;
  */

  // A is diagonal
  for(int i = im_point.x - width/2; i <= im_point.x + width/2; ++i)
  {
    for(int j = im_point.y - width/2; i <= im_point.x + width/2; ++i)
    {
      A[j*img_width+i][j*img_width+i] += 1;
      b[j*img_width+i] = ref_image.at<unsigned char>(j,i);
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
    {
      for(int icol = 0; icol < image.size().width; ++icol)
      {
        im_one_channel.at<unsigned char>(irow, icol) = image.at<Vec3b>(irow, icol)[ichannel];
        ref_one_channel.at<unsigned char>(irow, icol) = image.at<Vec3b>(irow, icol)[ichannel];
      }
    }

    solve_one_channel(z, x, ref_one_channel, im_one_channel, width);

    // Combine the image of each channel to form the final one
    for(int irow = 0; irow < image.size().height; ++irow)
      for(int icol = 0; icol < image.size().width; ++icol)
        image.at<Vec3b>(irow, icol)[ichannel] = im_one_channel.at<unsigned char>(irow, icol);  
  }
}


