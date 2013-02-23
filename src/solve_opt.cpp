#include "solvers.h"
#include "common.h"
#include <stdexcept>

using namespace std;
using namespace alglib;

void solve_opt(const vector<Point>& z, const vector<Point>& x, const cv::Mat& ref_image, cv::Mat& image, int width)
{
  // Channel independence
  if (z.size() != x.size())
    throw invalid_argument("size x != size z");

  /*
  -----x----
  |
  y
  |
  */

  real_2d_array fmatrix;
  fmatrix.setlength(image.size().width, image.size().width);

  for(int i = 0; i < (int)z.size(); ++i)
  {
        
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
