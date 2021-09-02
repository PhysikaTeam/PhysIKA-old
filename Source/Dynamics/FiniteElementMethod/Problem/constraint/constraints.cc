/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: constraint implementation
 * @version    : 1.0
 */
#include "constraints.h"
namespace PhysIKA {
using namespace std;
using namespace Eigen;
hard_position_constraint::hard_position_constraint(const Matrix<double, -1, -1>& nods)
    : dim_(nods.size()), rd_(nods.rows()) {}

size_t hard_position_constraint::Nx() const
{
    return dim_;
}

size_t hard_position_constraint::Nf() const
{
    if (rd_ == 3)
        return 3 * fixed3d_.size();
    if (rd_ == 2)
        return 2 * fixed2d_.size();
}

int hard_position_constraint::Val(const double* x, double* val) const
{
    Map<const MatrixXd> X(x, rd_, Nx() / rd_);
    Map<MatrixXd>       V(val, rd_, Nf() / rd_);

    if (rd_ == 3)
    {
        size_t i = 0;
        for (auto& elem : fixed3d_)
        {
            const size_t pid = elem.first;
            V.col(i) += X.col(pid) - elem.second;
            ++i;
        }
        return 0;
    }

    if (rd_ == 2)
    {
        size_t i = 0;
        for (auto& elem : fixed2d_)
        {
            const size_t pid = elem.first;
            V.col(i) += X.col(pid) - elem.second;
            ++i;
        }
        return 0;
    }

    return __LINE__;
}

int hard_position_constraint::Jac(const double* x, const size_t off, vector<Triplet<double>>* jac) const
{
    if (rd_ == 3)
    {
        size_t i = 0;
        for (auto& elem : fixed3d_)
        {
            const size_t pid = elem.first;
            jac->push_back(Triplet<double>(off + 3 * i + 0, 3 * pid + 0, 1));
            jac->push_back(Triplet<double>(off + 3 * i + 1, 3 * pid + 1, 1));
            jac->push_back(Triplet<double>(off + 3 * i + 2, 3 * pid + 2, 1));
            ++i;
        }
        return 0;
    }

    if (rd_ == 2)
    {
        size_t i = 0;
        for (auto& elem : fixed2d_)
        {
            const size_t pid = elem.first;
            jac->push_back(Triplet<double>(off + 2 * i + 0, 2 * pid + 0, 1));
            jac->push_back(Triplet<double>(off + 2 * i + 1, 2 * pid + 1, 1));
            ++i;
        }
        return 0;
    }

    return __LINE__;
}

int hard_position_constraint::Hes(const double* x, const size_t off, vector<vector<Triplet<double>>>* hes) const
{
    return __LINE__;
}

}  // namespace PhysIKA
