/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: linear solver list
 * @version    : 1.0
 */
#ifndef PhysIKA_LINEAR_SOLVER
#define PhysIKA_LINEAR_SOLVER
#include <Eigen/Sparse>
namespace PhysIKA {

// template<typename T>
// using linear_solver_type = std::function<int(
//   const Eigen::SparseMatrix<T, Eigen::RowMajor>& A,
//   const T* b,
//   const Eigen::SparseMatrix<T, Eigen::RowMajor>& J,
//   const T* c,
//   T* solution)>;

/**
 * linear solver class
 *
 */
template <typename T>
class linear_solver
{
public:
    using VEC = Eigen::Matrix<T, -1, 1>;
    using SPM = Eigen::SparseMatrix<T, Eigen::RowMajor>;
    virtual ~linear_solver() {}
    virtual int solve(const SPM& A, const T* b, const SPM& J, const T* c, T* solution) const = 0;
};

/**
 * unconstrainted linear solver class
 *
 */
template <typename T>
class unconstrainted_linear_solver : public linear_solver<T>
{
    using SPM = Eigen::SparseMatrix<T, Eigen::RowMajor>;

public:
    virtual ~unconstrainted_linear_solver() {}
    virtual int solve(const SPM& A, const T* b, const SPM& J, const T* c, T* solution) const = 0;
};

/**
 * constrainted linear solver, KKT solver.
 *
 */
template <typename T>
class KKT : public linear_solver<T>
{
    using SPM = Eigen::SparseMatrix<T, Eigen::RowMajor>;

public:
    virtual ~KKT() {}
    virtual int solve(const SPM& A, const T* b, const SPM& J, const T* c, T* solution) const = 0;
};

}  // namespace PhysIKA
#endif
