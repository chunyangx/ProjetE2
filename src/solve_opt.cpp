#include "solvers.h"
#include "common.h"
#include <stdexcept>

using namespace std;
using namespace alglib;
using namespace cv;

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

  real_1d_array b;
  b.setlength(nb_pixels);

  for(int i = 0; i < (int)z.size(); ++i)
  {
        
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
  } 
}

int main()
{
  real_2d_array fmatrix;
  fmatrix.setlength(2,2);
  fmatrix[0][0] = 1;
  fmatrix[1][1] = 1;

  real_1d_array b;
  b.setlength(2);
  b[0] = 1;
  b[1] = 2;

  ae_int_t nrows = 2;
  ae_int_t ncols = 2;

  ae_int_t info = 1;
  densesolverlsreport rep;
  real_1d_array x;
  x.setlength(2);

  rmatrixsolvels(fmatrix, nrows, ncols, b, 0.0, info, rep, x);  
}
