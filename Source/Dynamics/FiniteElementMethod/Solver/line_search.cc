/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: linesearch method for optimization problem.
 * @version    : 1.0
 */
#include "line_search.h"
#include <iostream>

using namespace std;
using namespace Eigen;
namespace PhysIKA {

template <typename T, size_t dim_>
T line_search(const T& val_init, const T& down, const std::shared_ptr<Functional<T, dim_>>& energy, std::shared_ptr<dat_str_core<T, dim_>>& data, const T* const _xk, const T* const _pk)
{
    Eigen::Map<const Matrix<T, -1, 1>> xk(_xk, dim_ * data->get_dof());
    Eigen::Map<const Matrix<T, -1, 1>> pk(_pk, dim_ * data->get_dof());

    // cout << "[INFO] linesearch:" << endl;
    const T c = 1e-4, c2 = 0.9;

    T Val_upbound, Val_func, down_new = 0;

    Matrix<T, -1, 1> xk_post(xk.rows(), xk.cols());
    Matrix<T, -1, 1> gra_tmp(xk.rows(), xk.cols());

    auto cal_val = [&](const T alp) -> T {
        data->set_zero();
        xk_post = xk + alp * pk;
        energy->Val(xk_post.data(), data);
        return data->get_val();
    };

    auto cal_gra = [&]() {
        energy->Gra(xk_post.data(), data);
        gra_tmp = data->get_gra();
    };

    auto zoom = [&](T alpha_low, T alpha_high, T val_low) -> T {
        T      alpha_star = alpha_high;
        size_t count_j    = 1;
        do
        {
            T alpha_j = 0.5 * (alpha_low + alpha_high);
            T val_j   = cal_val(alpha_j);
            if (val_j > val_init + c * alpha_j * down || val_j > val_low)
            {
                alpha_high = alpha_j;
            }

            else
            {
                cal_gra();
                T deri = pk.dot(gra_tmp);
                if (fabs(deri) <= -c2 * down)
                {
                    alpha_star = alpha_j;
                    break;
                }
                if (deri * (alpha_high - alpha_low) >= 0)
                    alpha_high = alpha_low;
                alpha_low = alpha_j;
                val_low   = val_j;
            }
            ++count_j;

            if (count_j > 20)
            {
                alpha_star = alpha_low;
                break;
            }
        } while (1);

        return alpha_star;
    };
    T      val_now, val_before = val_init, alpha_now = 1, alpha_before = 0, alpha_fin, alpha_max = 2;
    size_t count = 1;
    do
    {

        val_now = cal_val(alpha_now);
        if (val_now > val_init + c * alpha_now * down || (val_now >= val_before && count > 1))
        {
            alpha_fin = zoom(alpha_before, alpha_now, val_before);
            break;
        }
        cal_gra();
        T deri = pk.dot(gra_tmp);
        if (fabs(deri) <= -c2 * down)
        {
            alpha_fin = alpha_now;
            break;
        }
        if (deri >= 0)
        {
            alpha_fin = zoom(alpha_now, alpha_before, val_now);
            break;
        }
        alpha_before = alpha_now;
        alpha_now    = 0.5 * (alpha_now + alpha_max);

        val_before = val_now;
        ++count;
    } while (1);
    cout << "line search alpha is " << alpha_fin << endl;
    return alpha_fin == 0 ? 1 : alpha_fin;
}
template double line_search<double, 3>(
    const double&                                 val_init,
    const double&                                 down,
    const std::shared_ptr<Functional<double, 3>>& energy,
    std::shared_ptr<dat_str_core<double, 3>>&     data,
    const double* const                           xk,
    const double* const                           pk);

template double line_search<double, 1>(
    const double&                                 val_init,
    const double&                                 down,
    const std::shared_ptr<Functional<double, 1>>& energy,
    std::shared_ptr<dat_str_core<double, 1>>&     data,
    const double* const                           xk,
    const double* const                           pk);

template float line_search<float, 3>(
    const float&                                 val_init,
    const float&                                 down,
    const std::shared_ptr<Functional<float, 3>>& energy,
    std::shared_ptr<dat_str_core<float, 3>>&     data,
    const float* const                           xk,
    const float* const                           pk);

template float line_search<float, 1>(
    const float&                                 val_init,
    const float&                                 down,
    const std::shared_ptr<Functional<float, 1>>& energy,
    std::shared_ptr<dat_str_core<float, 1>>&     data,
    const float* const                           xk,
    const float* const                           pk);
// template class line_search<double, 2>;
// template class line_search<float, 3>;
// template class line_search<float, 2>;
}  // namespace PhysIKA
