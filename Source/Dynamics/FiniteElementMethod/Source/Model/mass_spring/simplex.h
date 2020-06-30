#ifndef _SIMPLEX_
#define _SIMPLEX_
#include "myvector.h"

//a data structure for mass spring modeling 
template<typename T>
class simplex
{
 public:
  size_t index_vertex[4];
  int dim_simplex; // one/two/three dimension simplex 
  T vol; // in 1 2 3 d
  simplex(){}
  ~simplex(){}
  simplex(T vol,int dim_simplex,const size_t (&index_vertex)[4]);
};

template class simplex<double>;
template class simplex<float>;
#endif
