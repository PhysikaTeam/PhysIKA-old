#ifndef PhysIKA_PRECONDITIONER
#define PhysIKA_PRECONDITIONER
#include <iostream>
#include "Common/DEFINE_TYPE.h"

namespace PhysIKA{
template<typename T>
class preconditioner{
 public:
  ~preconditioner(){}
  virtual VEC<T> operator () (const VEC<T>& r) const = 0;
};

template<typename T>
using precond_type = std::shared_ptr<preconditioner<T>>;

//Use diag(A) to approximate A
template<typename T>
class diag_preconditioner: public  preconditioner<T>{
 public:
  diag_preconditioner(const SPM_R<T>& A):diag_of_A_(A.diagonal()){}

  // z = inv(diag(A)) * r
  VEC<T> operator()(const VEC<T>& r) const{
    exit_if(r.size() != diag_of_A_.size());
    VEC<T> res = VEC<T>::Zero(r.size());
    res.array() = r.array() / diag_of_A_.array();
    return res;
  }
 private:
  const VEC<T> diag_of_A_;
};


}
#endif
