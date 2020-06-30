#ifndef _VERTEX_
#define _VERTEX_

#include "head.h"
#include "eigen_head.h"
#include "myvector.h"

// a data structure for mass spring modeling 
template<typename T,size_t dim>
class vertex
{
 public:
  myvector<T,dim> location;
  myvector<T,dim> location_lastIteration;
  myvector<T,dim> location_lastlastIteration;
  myvector<T,dim> velocity;
  myvector<T,dim> location_original;
  myvector<T,dim> location_lastFrame;
  myvector<T,dim> velocity_lastFrame;
  myvector<T,dim> location_maybe;
  myvector<T,dim> force_external;
  T mass;
  int isFixed; //  1:fixed points 0:free points

  myvector<T,dim> normal;// for opengl shading test 
  vertex();
  vertex(const myvector<T,dim> &location);
  ~vertex();
};

template class vertex<double,3>;
template class vertex<float,3>;
#endif
