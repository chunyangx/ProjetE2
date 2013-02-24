#include "kmeans.h"
#include <highgui.h>
#include <iostream>
#include <cmath>
#include <vector>
#include "dataanalysis.h"
#include "common.h"

using namespace cv;
using namespace std;
using namespace alglib;


Node::Node() : _son(0,NULL), _points(0), _center(0)
{
}

Node::~Node()
{
  for(int i=0;i<_son.size();++i){
    delete _son[i];
  }
}

void kmeanTree(Node* node, real_2d_array xy, int min_size){
  if(xy.rows()>min_size){
    clusterizerstate s;
    kmeansreport rep;

    clusterizercreate(s);
    clusterizersetpoints(s, xy, 2);
    clusterizersetkmeanslimits(s, 5, 0);
    clusterizerrunkmeans(s, 4, rep);

    if(int(rep.terminationtype)==1){
      vector<int> count(4);
      for(int c=0;c<4;c++){
        for(int i=0;i<rep.npoints;++i){
          if(rep.cidx[i] == c){
            count[c]++;
          }
        }
      }

      if(count[0] && count[1] && count[2] && count[3]){
        for(int c=0;c<4;c++){
          Node*  nodeSon = new Node;
          node->_son.push_back(nodeSon);
        }

        vector<real_2d_array> clusters(4);

        for(int c=0;c<4;c++){
          for(int f=0;f<rep.nfeatures;++f){
            node->_son[c]->_center.push_back((double)rep.c[c][f]);
          }

          int id = 0;
          clusters[c].setlength(count[c],rep.nfeatures);
          for(int i=0;i<rep.npoints;++i){
            if(rep.cidx[i] == c){
              for(int f=0;f<rep.nfeatures;++f){
                clusters[c][id][f] = xy[i][f];
              }
              id++;
            }
          }
        }

        for(int c=0;c<4;c++){
          kmeanTree(node->_son[c],clusters[c],min_size);
        }
      }
    }
  }
}

vector<double> neighborhood_vect(const Point& pixel, const int& w, const Mat& image)
{
  vector<double> NH;
  for(int x=pixel.x-w/2; x<=pixel.x+w/2; x++)
  {
    for(int y=pixel.y-w/2; y<=pixel.y+w/2; y++){
      for(int k=0;k<3;k++){
        NH.push_back((double) image.at<Vec3b>(x,y)[k]);
      }
    }
  }
  return NH;
}

real_2d_array neighborhood_array(const vector<Point>& pixel, const int& w, const Mat& image)
{
  real_2d_array NH;
  NH.setlength(pixel.size(),(w+1)*(w+1)*3);
  for(int id=0;id<pixel.size();id++){
  int i = 0;
  for(int x=pixel[id].x-w/2; x<=pixel[id].x+w/2; x++)
  {
    for(int y=pixel[id].y-w/2; y<=pixel[id].y+w/2; y++){
      for(int k=0;k<3;k++){
        NH[id][i]=image.at<Vec3b>(x,y)[k];
        i++;
      }
    }
  }
}
  return NH;
}


double diff_vect(vector<double> v1, vector<double> v2){
  double res = 0;
  for(int i=0; i<v1.size(); i++){
    res += (v1[i]-v2[i])*(v1[i]-v2[i]);
  }
  return sqrt(res);
}

Node* find_node(const vector<double>& NH, Node* root){
  if(root->_son.size()==0){
    return root;
  }
  double minDist = numeric_limits<double>::max();
  double dist;
  int minC = -1;
  for(int c=0;c<4;c++){
    dist = diff_vect(NH,root->_son[c]->_center);

    if(dist<minDist){
      minDist = dist;
      minC = c;
    }
  }
  return find_node(NH,root->_son[minC]);
}

void fill_tree(const vector<Point>& pixel, const int& w, const Mat& image, Node* root){
  for(int i=0;i<pixel.size();i++){
    printf("%d\n",i);
    vector<double> NH = neighborhood_vect(pixel[i], w, image);
    Node* node = find_node(NH, root);
    node->_points.push_back(pixel[i]);
  }
}


Node* constructTree(const vector<Point>& pixel, const int& w, const Mat& image)
{
  real_2d_array NH = neighborhood_array(pixel, w, image);
  Node* root = new Node;
  kmeanTree(root, NH, pixel.size()/100);
  fill_tree(pixel,w,image,root);

  return root;
}

Point findTreeNNH(const Point& p, const int& w, const Mat& imagea, const Mat& imageb, Node* root)
{
  vector<double> NH = neighborhood_vect(p, w, imagea);
  Node* node = find_node(NH, root);

  return nearestNH(p, w, imagea, node->_points, imageb);
}

void findTreeNNH(const vector<Point>& pixel, const int& w, const Mat& imagea, const Mat& imageb, Node* root, vector<Point>& NNH)
{
  NNH.clear();
  for(int i=0;i<pixel.size();i++){
    vector<double> NH = neighborhood_vect(pixel[i], w, imagea);
    Node* node = find_node(NH, root);
    NNH.push_back(nearestNH(pixel[i], w, imagea, node->_points, imageb));
  }
}


int test_kmeans(int argc, char** argv){

  Mat image;
  image = imread(argv[1], 1);

  Mat imagebis;
  imagebis = imread(argv[2], 1);


  vector<Point> pixel_grid;
  allPoints(pixel_grid, 10, image);

  Node* root2 = constructTree(pixel_grid,10,image);

  vector<Point> pixel_grid_bis;
  grid(pixel_grid_bis, 10, imagebis);

  vector<Point> NNH;
  findTreeNNH(pixel_grid_bis, 10, imagebis, image, root2, NNH);

}