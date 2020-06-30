#ifndef _OBJECT_CREATOR_
#define _OBJECT_CREATOR_

#include "head.h"
#include "myvector.h"
#include "simplex.h"
#include "mass_spring_obj.h"

template<typename T>
class object_creator /*: public Physika::Module*/ //create the input mesh: all are the simplex of one to three dimension
{
 public:
  std::string out_dir;
  T length,width,height;
  T dmetric;
  int dim_simplex;
  int*** index_for_vertex; //因为需要赋值为-1,所以用int 
  size_t lc,wc,hc; //节点数
  
  std::string object_name;
  object_creator();
  ~object_creator();
  int create_object();
};

template class object_creator<double>;
template class object_creator<float>;
#endif

