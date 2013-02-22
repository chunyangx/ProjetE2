#include "solvers.h"

using namespace std;
using namespace alglib;

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
