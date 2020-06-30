#ifndef _IO_
#define _IO_
#include "head.h"
#include "vertex.h"
#include "simplex.h"
#include "mass_spring_obj.h"

// for simple vtk file format's input and output testing and get constraints for vertexs from csv file format
// not a computation module 
template<typename T>
class io
{
 public:
  io(){}
  ~io(){}
  //name 不用引用，因为可能实参直接是"xxx" 
  int saveAsVTK(mass_spring_obj<T,3>* my_mass_spring_obj,const std::string name);
  int getConstraintFromCsv(mass_spring_obj<T,3>* my_mass_spring_obj,const std::string name);
  int getVertexAndSimplex(std::vector<vertex<T,3 > > &myvertexs,std::vector<simplex<T> > &mysimplexs,int &dim_simplex,const std::string name);
  int saveTimeNormPair(mass_spring_obj<T,3 >* my_mass_spring_obj,const std::string name);
};

template class io<double>;
template class io<float>;
#endif
