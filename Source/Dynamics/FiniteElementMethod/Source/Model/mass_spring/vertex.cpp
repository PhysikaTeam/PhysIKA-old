#include "vertex.h"
#include "autodiff.h"
using namespace std;
using namespace Eigen;
//DECLARE_DIFFSCALAR_BASE();

template<typename T,size_t dim>
vertex<T,dim>::vertex()
{
  ;
}

template<typename T,size_t dim>
vertex<T,dim>::vertex(const myvector<T,dim> &location)
{
  this->location=location;
  this->location_original=location;
  this->location_maybe=location;
  this->velocity=myvector<T,dim>();
  this->force_external=myvector<T,dim>();
  isFixed=0; // all free when at beginning
}

template<typename T,size_t dim>
vertex<T,dim>::~vertex()
{
  ;
}
