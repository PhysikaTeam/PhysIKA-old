#ifndef _SIMULATOR_
#define _SIMULATOR_

#include "head.h"
#include "mass_spring_obj.h"

//It here is regarded as a computation module for Physika's system design. It here includes a mass_spring_obj asnode's subclass.  
template<typename T>
class simulator /*: public Physika::Module*/
{
 public:
  std::string simulation_type;
  size_t frame;
  T dt;
  std::string force_function;

  T gravity;
  T density;
  T stiffness;

  mass_spring_obj<T,3>* test_mass_spring_obj;
  std::string newton_fastMS; // newton  fastMS
  // if is newton, with a line search strategy
  int line_search;
  T weight_line_search;
  
  std::string out_dir; 
  simulator();
  ~simulator();

  int simulate(int i);
};

template class simulator<double>;
template class simulator<float>;
#endif
