/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: tensor class definition
 * @version    : 1.0
 */
#ifndef PhysIKA_TENSOR
#define PhysIKA_TENSOR
#include <Eigen/Dense>
#include <array>
#include <iostream>
namespace PhysIKA {
using namespace std;

/**
 * a simple 4th tensor class.
 *
 */
template <typename T, size_t i, size_t j, int k, int l>
class fourth_tensor
{
    using tensor_type = std::array<std::array<Eigen::Matrix<T, k, l>, j>, i>;

public:
    fourth_tensor()
    {
        for (size_t i_id = 0; i_id < i; ++i_id)
        {
            for (size_t j_id = 0; j_id < j; ++j_id)
            {
                tensor_[i_id][j_id] = Eigen::Matrix<T, k, l>::Zero();
            }
        }
    }

    const Eigen::Matrix<T, k, l>& operator()(const size_t& row, const size_t& col) const
    {
        return tensor_[row][col];
    }

    Eigen::Matrix<T, k, l>& operator()(const size_t& row, const size_t& col)
    {
        return tensor_[row][col];
    }

    void Flatten(Eigen::Matrix<T, i * j, k * l>& flatted)
    {
        flatted.setZero();
        for (size_t row_out = 0; row_out < i; ++row_out)
        {
            for (size_t col_out = 0; col_out < j; ++col_out)
            {
                Eigen::Map<Eigen::Matrix<T, 1, k * l>> vec(tensor_[row_out][col_out].data());
                flatted.row(col_out * i + row_out) = vec;
            }
        }
    }

    tensor_type tensor_;
};

template <typename T, size_t i, size_t j, int k, int l>
fourth_tensor<T, i, j, k, l> operator+(const fourth_tensor<T, i, j, k, l>& lhs, const fourth_tensor<T, i, j, k, l>& rhs)
{
    fourth_tensor<T, i, j, k, l> res;
#pragma omp parallel for
    for (size_t res_row = 0; res_row < i; ++res_row)
    {
        for (size_t res_col = 0; res_col < j; ++res_col)
        {
            res(res_row, res_col) = lhs(res_row, res_col) + rhs(res_row, res_col);
        }
    }
    return res;
}

template <typename T, size_t i, size_t j, int k, int l>
fourth_tensor<T, i, j, k, l> operator*(const fourth_tensor<T, i, j, k, l>& lhs, const Eigen::Matrix<T, k, l>& rhs)
{
    fourth_tensor<T, i, j, k, l> res;
    for (size_t res_row = 0; res_row < i; ++res_row)
    {
        for (size_t res_col = 0; res_col < j; ++res_col)
        {
            for (size_t e_id = 0; e_id < i; ++e_id)
            {
                res(res_row, res_col) += lhs(res_row, e_id) * rhs(e_id, res_col);
            }
        }
    }
    return res;
}

template <typename T, size_t i, size_t j, int k, int l>
fourth_tensor<T, i, j, k, l> operator*(const Eigen::Matrix<T, k, l>& lhs, const fourth_tensor<T, i, j, k, l>& rhs)
{
    fourth_tensor<T, i, j, k, l> res;
    for (size_t res_row = 0; res_row < i; ++res_row)
    {
        for (size_t res_col = 0; res_col < j; ++res_col)
        {
            for (size_t e_id = 0; e_id < i; ++e_id)
            {
                res(res_row, res_col) += lhs(res_row, e_id) * rhs(e_id, res_col);
            }
        }
    }
    return res;
}

template <typename T, size_t i, size_t j, int k, int l>
ostream& operator<<(ostream& os, const fourth_tensor<T, i, j, k, l>& ten)
{

    os << i << " " << j << " " << k << " " << l << endl;
    os << "--------------------------------------------------------------------" << endl;
    for (size_t row_out = 0; row_out < i; ++row_out)
    {
        for (size_t row_inner = 0; row_inner < k; ++row_inner)
        {
            os << "|";
            for (size_t col_out = 0; col_out < j; ++col_out)
            {
                for (size_t col_inner = 0; col_inner < l; ++col_inner)
                {
                    printf("%+05.2f ", (ten(row_out, col_out))(row_inner, col_inner));
                }
                os << "|";
            }
            os << endl;
        }
        os << "--------------------------------------------------------------------" << endl;
    }
}

}  // namespace PhysIKA
#endif
