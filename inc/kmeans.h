#ifndef KMEANS_H
#define KMEANS_H
#endif

#include <cv.h>
#include <vector>
#include "dataanalysis.h"

using namespace cv;
using namespace std;
using namespace alglib;

class Node{
  public:
  Node();
  ~Node();

  vector<Node*> _son;
  vector<double> _center; 
  vector<Point> _points; // Nonempty only for leaf nodes.
};

void kmean();

// Construct a k-mean tree with root at node given in parameters.
void kmeanTree(Node* node, real_2d_array xy, int min_size);

// Return the neighborhood of a pixel p in a given image into a vector of double.
vector<double> neighborhood_vect(const Point& pixel, const int& w, const Mat& image);

// Return the neighborhoods of a set of points in a 2d array.
real_2d_array neighborhood_array(const vector<Point>& pixel, const int& w, const Mat& image);

// L2 distance between two vector of double.
double diff_vect(vector<double> v1, vector<double> v2);

// Find the leaf node of the neighborhood NH in the tree.
Node* find_node(const vector<double>& NH, Node* root);

// Fill the leaves of the tree with given points.
void fill_tree(const vector<Point>& pixel, const int& w, const Mat& image, Node* root);

// Construct a tree for a set of neighborhood (related to the points in the vector pixel).
Node* constructTree(const vector<Point>& pixel, const int& w, const Mat& image);

// Find the nearest neighborhood of a given one using a tree.
Point findTreeNNH(const Point& p, const int& w, const Mat& imagea, const Mat& imageb, Node* root);

// Find the nearest neighborhoods of a given set using a tree.
void findTreeNNH(const vector<Point>& pixel, const int& w, const Mat& imagea, const Mat& imageb, Node* root, vector<Point>& NNH);