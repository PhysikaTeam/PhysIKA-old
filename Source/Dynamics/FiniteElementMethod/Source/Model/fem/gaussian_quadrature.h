/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: gaussian quadratures for finite element method.
 * @version    : 1.0
 */
#ifndef FEM_QUADRATURE
#define FEM_QUADRATURE
#include <array>
#include <Eigen/Core>

namespace PhysIKA {
/**
 * gauss base interface
 *
 */
template <typename T, size_t num_qdrt_per_axis>
struct gauss_base
{
    static std::array<T, num_qdrt_per_axis> P_;
    static std::array<T, num_qdrt_per_axis> W_;
};

template <typename T>
struct gauss_base<T, 1>
{
    static std::array<T, 1> P_;
    static std::array<T, 1> W_;
};

template <typename T>
std::array<T, 1> gauss_base<T, 1>::P_ = { 0.0 };
template <typename T>
std::array<T, 1> gauss_base<T, 1>::W_ = { 2.0 };

template <typename T>
struct gauss_base<T, 2>
{
    static std::array<T, 2> P_;
    static std::array<T, 2> W_;
};

template <typename T>
std::array<T, 2> gauss_base<T, 2>::P_ = { -0.5773502691896258, 0.5773502691896258 };

template <typename T>
std::array<T, 2> gauss_base<T, 2>::W_ = { 1.0, 1.0 };

/**
 * gaussian quadratures
 *
 */
template <typename T, size_t dim_, size_t qdrt_axis_, size_t verts_cell_>
class quadrature
{
public:
    Eigen::Matrix<T, dim_, -1> PNT_;
    std::vector<T>             WGT_;
    quadrature()
    {
        const size_t qdrt_num = static_cast<size_t>(pow(qdrt_axis_, dim_));
        PNT_.resize(dim_, qdrt_num);
        WGT_.resize(qdrt_num, 1.0);
        PNT_.setZero();
        std::vector<size_t> idx;
        init(idx, 0);
    }

private:
    void init(std::vector<size_t>& idx, size_t PNT_id)
    {
        const size_t depth = idx.size();
        if (idx.size() == dim_)
        {
            for (size_t i = 0; i < dim_; ++i)
            {
                PNT_(i, PNT_id) = gauss_base<T, qdrt_axis_>::P_[idx[i]];
                WGT_[PNT_id] *= gauss_base<T, qdrt_axis_>::W_[idx[i]];
            }
        }
        else
        {
            for (size_t i = 0; i < qdrt_axis_; ++i)
            {
                auto idx_next = idx;
                idx_next.push_back(i);
                init(idx_next, PNT_id + pow(2, depth) * i);
            }
        }
        return;
    }
};

}  // namespace PhysIKA
#endif
