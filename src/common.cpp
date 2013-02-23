#include "common.h"
#include <cmath>

using namespace cv;

double diff(const Vec3b& pa, const Vec3b& pb)
{
  double res = 0;
  for(int i = 0; i < 3; ++i)
    res += (pa[i]-pb[i])*(pa[i]-pb[i]);
  return sqrt(res);
}




