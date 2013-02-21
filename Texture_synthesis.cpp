#include <cv.h>
#include <highgui.h>

using namespace cv;

int main(int argc, char** argv)
{
  Mat image;
  image = imread(argv[1], 1);

  if( argc != 2 || !image.data )
  {
    printf( "No image data \n" );
    return -1;
  }

  namedWindow( "Display Image", 256);
  imshow( "Display Image", image );

  Mat syn_image(256,256, CV_8UC3);
  waitKey();

  return 0;

}
