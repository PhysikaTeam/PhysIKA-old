#ifndef SEMI_WRAPPER_JJ_H
#define SEMI_WRAPPER_JJ_H

#include "Solver/semi_implicit_euler.h"

template<typename T>
class semi_wrapper
{
public:
  semi_wrapper(): semi_implicit_(nullptr) { }
  virtual std::shared_ptr<semi_implicit<T>> get_semi_implicit() const { return semi_implicit_;}
  
protected:
  std::shared_ptr<semi_implicit<T>> semi_implicit_;
};


#endif // SEMI_WRAPPER_JJ_H
