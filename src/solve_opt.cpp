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

  vector<double> X(nb_pixels);
  vector<double> Z(nb_pixels);

  // gaussian width
  double gamma = width/2;

  // assign gaussian weights
  for(int k = 0; k < x.size(); ++k)
  {
    for(int i = x[k].x - width/2; i < x[k].x + width/2; ++i)
    {
      for(int j = x[k].y - width/2; j < x[k].y + width/2; ++j)
      {
        int x_NH = i - x[k].x;
        int y_NH = j - x[k].y;
        double gaussian_weight = exp(-(x_NH*x_NH+y_NH*y_NH)/(2*gamma*gamma));
        X[i*image_height+j] += gaussian_weight;
        Z[i*image_height+j] += gaussian_weight*ref_image.at<unsigned char>(z[k].x +i-x[k].x,z[k].y+j-x[k].y);
      }
    }
  }

  for(int i = 0; i < image.size().width; ++i)
  {
    for(int j = 0; j < image.size().height; ++j)
    {
      int k = image_height*i+j;
      if(X[k] != 0){
        image.at<unsigned char>(j,i) = Z[k]/X[k];
      }
    }
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
    for(int irow = 0; irow < image.size().height; ++irow)
      for(int icol = 0; icol < image.size().width; ++icol)
        image.at<Vec3b>(irow, icol)[ichannel] = im_one_channel.at<unsigned char>(irow, icol);  
  }
}

void wsolve_one_channel_bis(const vector<Point>& z, const vector<Point>& x, const Mat& ref_image, Mat& image, int width, const vector<double>& weights){
  int nb_pixels = image.size().width*image.size().height;
  int image_height = image.size().height;
  vector<double> X(nb_pixels, 0);
  vector<double> Z(nb_pixels, 0);
  double gamma = width/2;

  for(int k = 0; k < x.size(); ++k)
  {
    for(int i = x[k].x - width/2; i < x[k].x + width/2; ++i)
    {
      for(int j = x[k].y - width/2; j < x[k].y + width/2; ++j)
      {
        int x_NH = i - x[k].x;
        int y_NH = j - x[k].y;
        double gaussian_weight = exp(-(x_NH*x_NH+y_NH*y_NH)/(2*gamma*gamma));
        X[i*image_height+j] += weights[k]*gaussian_weight;
        Z[i*image_height+j] += weights[k]*gaussian_weight*ref_image.at<unsigned char>(z[k].x +i-x[k].x,z[k].y+j-x[k].y);
      }
    }
  }

  for(int i = 0; i < image.size().width; ++i)
  {
    for(int j = 0; j < image.size().height; ++j)
    {
      int k = image_height*i+j;
      if(X[k] != 0){
        image.at<unsigned char>(j,i) = Z[k]/X[k];
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
    for(int irow = 0; irow < image.size().height; ++irow)
      for(int icol = 0; icol < image.size().width; ++icol)
        image.at<Vec3b>(irow, icol)[ichannel] = im_one_channel.at<unsigned char>(irow, icol);  
  }
}

void solve_one_channel_grad(const vector<Point>& z, const vector<Point>& x, const Mat& ref_image, Mat& image, int width, const vector<double>& weights)
{
  /*
  -----x----
  |
  y
  |
  */

  printf("Optmize\n");
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

    double mu = .1;
    for(int irow = x[k].x - width/2 +1; irow < x[k].x + width/2 -1; ++irow)
    {
      for(int icol = x[k].y - width/2 +1; icol < x[k].y + width/2 -1; ++icol)
      {
        /*int x_NH = i - x[k].x;
        int y_NH = j - x[k].y;
        double gaussian_weight = exp(-(x_NH*x_NH+y_NH*y_NH)/(2*gamma*gamma));*/
        double gaussian_weight = 1.;


        sparseadd(A, icol*image_rows+irow+1, icol*image_rows+irow+1, mu*gaussian_weight);
        sparseadd(A, icol*image_rows+irow+1, icol*image_rows+irow-1, -mu*gaussian_weight);
        b[icol*image_rows+irow+1] += mu*gaussian_weight*(ref_image.at<unsigned char>(z[k].x + (irow-x[k].x)+1, z[k].y + (icol-x[k].y)) - ref_image.at<unsigned char>(z[k].x + (irow-x[k].x)-1, z[k].y + (icol-x[k].y)));
        
        sparseadd(A, icol*image_rows+irow-1, icol*image_rows+irow-1, mu*gaussian_weight);
        sparseadd(A, icol*image_rows+irow-1, icol*image_rows+irow+1, -mu*gaussian_weight);
        b[icol*image_rows+irow-1] += mu*gaussian_weight*(ref_image.at<unsigned char>(z[k].x + (irow-x[k].x)-1, z[k].y + (icol-x[k].y)) - ref_image.at<unsigned char>(z[k].x + (irow-x[k].x)+1, z[k].y + (icol-x[k].y)));
        
      }
    }



  }

  //print_matrix(A, 20);
  //print_array(b, 10);

  //printf("startconvert");

  sparseconverttocrs(A);
  linlsqrstate s;
  linlsqrreport rep;
  real_1d_array sol;
  sol.setlength(nb_pixels);
  init_1d_array(sol, nb_pixels);

  // solve the problem
  linlsqrcreate(nb_pixels, nb_pixels, s);
  //printf("startsolve");
  linlsqrsolvesparse(s, A, b);
  linlsqrresults(s, sol, rep);
  //printf("endsolve");

  // print_array(sol, 20);
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

void solve_opt_grad(const std::vector<cv::Point>& z, const std::vector<cv::Point>& x, const cv::Mat& ref_image, cv::Mat& image, int width, const std::vector<double>& weights)
{
  printf("Optmize_grad\n");
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

    solve_one_channel_grad(z, x, ref_one_channel, im_one_channel, width, weights);
    
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

void solve_basic(const std::vector<cv::Point>& z, const std::vector<cv::Point>& x, const cv::Mat& ref_image, cv::Mat& image, int width)
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

    solve_one_channel_basic(z, x, ref_one_channel, im_one_channel, width);
    
    // Combine the image of each channel to form the final one
    for(int irow = 0; irow < image.rows; ++irow)
      for(int icol = 0; icol < image.cols; ++icol)
        image.at<Vec3b>(irow, icol)[ichannel] = im_one_channel.at<unsigned char>(irow, icol);  
  }
}

