/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: customized precondition conjugate gradient method
 * @version    : 1.0
 */
#ifndef PhysIKA_PCG
#define PhysIKA_PCG

#include <memory>

#include "Common/DEFINE_TYPE.h"
#include "Common/config.h"

#include "preconditioner.h"

namespace PhysIKA {
/**
 * precondition conjugate gradient method.
 *
 */
template <typename T, typename SPM_TYPE>
class PCG
{
public:
    PCG(const precond_type<T>& M = nullptr, const T tol = 1e-10, const bool sym = false);
    int solve2(const SPM_TYPE& A, const T* b, const SPM_TYPE& J, const T* c, T* solution)
    {
        return solve(A, b, solution);
    }
    int solve(const SPM_TYPE& A, const T* b, T* solution);

private:
    size_t                max_itrs_;
    double                tol_;
    const precond_type<T> M_;
    const bool            sym_;
};

}  // namespace PhysIKA
#endif
