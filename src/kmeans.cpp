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

//  Implementation of a k-mean tree structure to perform an approach NN search.

Node::Node() : _son(0,NULL), _points(0), _center(0)
{
}

Node::~Node()
{
  for(int i=0;i<_son.size();++i){
    delete _son[i];
  }
}

Node* Node::getSon(int i){
  return _son[i];
}

void Node::addSon(Node* node){
  _son.push_back(node);
}

void Node::addCenterCoord(double d){
  _center.push_back(d);
}

void Node::addPoint(Point p){
  _points.push_back(p);
}

int Node::numberSon(){
  return (int) _son.size();
}

vector<double>* Node::getCenter(){
  return &_center;
}

vector<Point>* Node::getPoints(){
  return &_points;
}

// Construction of a kmean Trea for the neighborhoods stored in xy.
void kmeanTree(Node* node, real_2d_array xy, int min_size){
  if(xy.rows()>min_size){ //Stop if we have less than min_size neighborhoods.

    // K_mean (K=4) :
    clusterizerstate s;
    kmeansreport rep;

    clusterizercreate(s);
    clusterizersetpoints(s, xy, 2);
    clusterizersetkmeanslimits(s, 5, 0);
    clusterizerrunkmeans(s, 4, rep);

    if(int(rep.terminationtype)==1){ // Check if the K_mean step finished correctly.

      // Count the number of neighborhoods in each cluster.
      vector<int> count(4);
      for(int c=0;c<4;c++){
        for(int i=0;i<rep.npoints;++i){
          if(rep.cidx[i] == c){
            count[c]++;
          }
        }
      }

      if(count[0] && count[1] && count[2] && count[3]){ // Check that each cluster is non-empty.

        // Create new nodes
        for(int c=0;c<4;c++){
          Node*  nodeSon = new Node;
          node->addSon(nodeSon);
        }

        vector<real_2d_array> clusters(4);

        for(int c=0;c<4;c++){
          // Add the center element of each cluster in its related node.
          for(int f=0;f<rep.nfeatures;++f){
            node->getSon(c)->addCenterCoord((double)rep.c[c][f]);
          }

          // Separate the neighborhoods into the new nodes.
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

        // Iterate the construction on the new nodes.
        for(int c=0;c<4;c++){
          kmeanTree(node->getSon(c),clusters[c],min_size);
        }
      }
    }
  }
}

// Return the neighborhood related to a pixel and a width w in image.
vector<double> neighborhood_vect(const Point& pixel, const int& w, const Mat& image)
{
  vector<double> NH;
  for(int x=pixel.x-w/2; x<pixel.x+w/2; x++)
  {
    for(int y=pixel.y-w/2; y<pixel.y+w/2; y++){
      for(int k=0;k<3;k++){
        NH.push_back((double) image.at<Vec3b>(x,y)[k]);
      }
    }
  }
  return NH;
}

// Create the array NH which contains the neighborhoods of each point in pixel in image.
real_2d_array neighborhood_array(const vector<Point>& pixel, const int& w, const Mat& image)
{
  real_2d_array NH;
  NH.setlength(pixel.size(),(w+1)*(w+1)*3);
  for(int id=0;id<pixel.size();id++){
  int i = 0;
  for(int x=pixel[id].x-w/2; x<pixel[id].x+w/2; x++)
  {
    for(int y=pixel[id].y-w/2; y<pixel[id].y+w/2; y++){
      for(int k=0;k<3;k++){
        NH[id][i]=image.at<Vec3b>(x,y)[k];
        i++;
      }
    }
  }
}
  return NH;
}

// Return the difference of two vectors of doubles.
double diff_vect(vector<double> v1, vector<double> v2){
  double res = 0;
  for(int i=0; i<v1.size(); i++){
    res += (v1[i]-v2[i])*(v1[i]-v2[i]);
  }
  return sqrt(res);
}

// Find the leaf node in the tree (root) which should countains the neighborhood NH.
Node* find_node(const vector<double>& NH, Node* root){
  if(root->numberSon()==0){
    return root;
  }
  double minDist = numeric_limits<double>::max();
  double dist;
  int minC = -1;
  for(int c=0;c<4;c++){
    dist = diff_vect(NH,*root->getSon(c)->getCenter());

    if(dist<minDist){
      minDist = dist;
      minC = c;
    }
  }
  return find_node(NH,root->getSon(minC));
}

// Fill the leaves of the tree (root) with the neighborhoods of each point in pixel.
void fill_tree(const vector<Point>& pixel, const int& w, const Mat& image, Node* root){
  for(int i=0;i<pixel.size();i++){
    vector<double> NH = neighborhood_vect(pixel[i], w, image);
    Node* node = find_node(NH, root);
    node->addPoint(pixel[i]);
  }
}

// Build a kmeans tree for an image and a set of points.
Node* constructTree(const vector<Point>& pixel, const int& w, const Mat& image)
{
  real_2d_array NH = neighborhood_array(pixel, w, image);
  Node* root = new Node;
  kmeanTree(root, NH, pixel.size()/100);
  fill_tree(pixel,w,image,root);

  return root;
}

// Find the nearest neighborhood of a point p in imagea using the tree (root) related to imageb.
Point findTreeNNH(const Point& p, const int& w, const Mat& imagea, const Mat& imageb, Node* root)
{
  vector<double> NH = neighborhood_vect(p, w, imagea);
  Node* node = find_node(NH, root);

  return nearestNH(p, w, imagea, *node->getPoints(), imageb);
}

// Same as above for a set of points.
void findTreeNNH(const vector<Point>& pixel, const int& w, const Mat& imagea, const Mat& imageb, Node* root, vector<Point>& NNH)
{
  NNH.clear();
  for(int i=0;i<pixel.size();i++){
    vector<double> NH = neighborhood_vect(pixel[i], w, imagea);
    Node* node = find_node(NH, root);
    NNH.push_back(nearestNH(pixel[i], w, imagea, *node->getPoints(), imageb));
  }
}