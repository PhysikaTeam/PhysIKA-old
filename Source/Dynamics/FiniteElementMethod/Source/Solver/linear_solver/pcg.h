/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: precondition conjugate gradient method
 * @version    : 1.0
 */
#ifndef PhysIKA_PCG_NEW
#define PhysIKA_PCG_NEW
#include <Eigen/Sparse>

#include <iostream>
#include "Common/DEFINE_TYPE.h"
#include "linear_solver.h"

namespace PhysIKA {

template <typename T>
using SPM = Eigen::SparseMatrix<T, Eigen::RowMajor>;

/**
 * eigen-impl preconditioner conjugate gradient
 *
 */
template <typename T>
class EIGEN_PCG : public unconstrainted_linear_solver<T>
{
public:
    EIGEN_PCG(const bool hes_is_constant, const T tol = 1e-3)
        : hes_is_constant_(hes_is_constant), tol_(tol) {}

    int solve(const Eigen::SparseMatrix<T, Eigen::RowMajor>& A, const T* b, const Eigen::SparseMatrix<T, Eigen::RowMajor>& J, const T* c, T* solution) const
    {
        using namespace std;
        assert(J.rows() == 0);
        const size_t                              total_dim = A.cols();
        Eigen::Map<const Eigen::Matrix<T, -1, 1>> rhs(b, total_dim);
        Eigen::Map<Eigen::Matrix<T, -1, 1>>       sol(solution, total_dim);

        Eigen::ConjugateGradient<Eigen::SparseMatrix<T, Eigen::RowMajor>, Eigen::Lower | Eigen::Upper, Eigen::DiagonalPreconditioner<T>> cg;
        //    cg.setMaxIterations(sqrt(double(total_dim))/4);
        cg.setTolerance(tol_);
        cg.compute(A);
        sol = cg.solve(rhs);
        std::cout << "norm of residual of linear equation: " << (A * sol - rhs).norm() << std::endl
                  << "iteratoin: " << cg.iterations() << std::endl;

        if (cg.info() == Eigen::Success)
            return 0;
        else if (cg.info() == Eigen::NoConvergence)
            std::cout << "Solve linear equation fail: Not Converge." << std::endl;
        else if (cg.info() == Eigen::NumericalIssue)
            std::cout << "Solve linear equation fail: The provided data did not satisfy the prerequisites." << std::endl;
        else
            std::cout << "Solve linear equation fail: The inputs are invalid, or the algorithm has been improperly called. When assertions are enabled, such errors trigger an assert." << std::endl;
        return __LINE__;
    }

private:
    const bool hes_is_constant_;
    const T    tol_;
};

}  // namespace PhysIKA
#endif
