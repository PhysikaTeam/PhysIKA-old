#ifndef PhysIKA_SPM_EIG_JAC
#define PhysIKA_SPM_EIG_JAC
#include "Common/DEFINE_TYPE.h"
namespace PhysIKA{
template<typename T>
int eig_jac(const SPM_R<T>& mat_A, MAT<T>&eig_vec, VEC<T>& eig_val);
}
#endif
