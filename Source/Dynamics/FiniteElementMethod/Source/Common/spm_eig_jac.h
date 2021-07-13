/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: eigenvalues for sparse matrix jacobian
 * @version    : 1.0
 */
#ifndef PhysIKA_SPM_EIG_JAC
#define PhysIKA_SPM_EIG_JAC
#include "Common/DEFINE_TYPE.h"
namespace PhysIKA {
template <typename T>
int eig_jac(const SPM_R<T>& mat_A, MAT<T>& eig_vec, VEC<T>& eig_val);
}
#endif
