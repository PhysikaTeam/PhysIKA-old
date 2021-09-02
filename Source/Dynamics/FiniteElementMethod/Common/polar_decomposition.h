/**
 * @author     : LamKamhang (Cool_Lam@outlook.com)
 * @date       : 2021-04-30
 * @description: polar decomposition.
 * @version    : 1.0
 */
#ifndef PhysIKA_POLAR_DECOMPOSITION_H
#define PhysIKA_POLAR_DECOMPOSITION_H

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Sparse>

#include "Common/DEFINE_TYPE.h"

namespace PhysIKA {
/**
   *  @todo current version does not support other dim.
   *
   */
template <typename T, typename Mat, size_t dim_ = 3>
int polar_decomposition(
    const Mat& A,
    Mat&       R,
    size_t     max_it = 5,
    T          tol    = scalar_eps<T>())
{
    // error_msg_ext_cond(
    //     A.cols() != dim_ || A.rows() != dim_,
    //     "A is a %lux%lu matrix, not a 3x3", dim_, dim_);

    // // current version only supports dim == 3.
    // error_msg_ext_cond(
    //     dim_ != 3,
    //     "current polar decomposition does not support other dim<%lu>, only 3",
    //     dim_);

    // simply copy the origin polar decomposition.
    Eigen::Quaternion<T> q(A);
    q.normalize();
    for (size_t i = 0; i < max_it; ++i)
    {
        const Mat&                R = q.matrix();
        Eigen::Matrix<T, dim_, 1> omega =
            (R.col(0).cross(A.col(0)) + R.col(1).cross(A.col(1)) + R.col(2).cross(A.col(2))) * (1.0 / fabs(R.col(0).dot(A.col(0)) + R.col(1).dot(A.col(1)) + R.col(2).dot(A.col(2))) + tol);
        T w = omega.norm();
        if (w <= tol)
        {
            break;
        }
        q = Eigen::Quaternion<T>(Eigen::AngleAxis<T>(w, (1.0 / w) * omega)) * q;
        q.normalize();
    }
    R = q.matrix();
    return 0;
}
}  // namespace PhysIKA

#endif /* PhysIKA_POLAR_DECOMPOSITION_H */
