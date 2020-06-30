#include "simplex.h"

template<typename T>
simplex<T>::simplex(T vol,int dim_simplex,const size_t (&index_vertex)[4])
{
  this->dim_simplex=dim_simplex;
  this->vol=vol;
  size_t i;
  for(i=0;i<4;++i)
    {
      this->index_vertex[i]=index_vertex[i];
    }
}
