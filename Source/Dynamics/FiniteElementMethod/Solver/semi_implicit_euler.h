/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: semi implicit method for differential equation.
 * @version    : 1.0
 */
#ifndef SEMI_IMPLICIT_EUELER_JJ_H
#define SEMI_IMPLICIT_EUELER_JJ_H

#include <Eigen/Core>
#include <unsupported/Eigen/KroneckerProduct>
#include <iostream>

using namespace std;

/**
 * semi implicit method for solving a differential equation
 *
 */
template <typename T>
class semi_implicit
{
public:
    semi_implicit(T h, const Eigen::Matrix<T, -1, 1>& m, const Eigen::Matrix<T, -1, 1>& x)
        : mass_(m), x_(x), h_(h)
    {
        mass_ = mass_.cwiseInverse();
        mass_ = Eigen::kroneckerProduct(mass_, Eigen::Matrix<T, 3, 1>::Ones()).eval();
        v_    = Eigen::Matrix<T, -1, 1>::Zero(mass_.size());
    }
    void update_status(const Eigen::Matrix<T, -1, 1>& x)
    {
        x_ = x;
        // v_ = v;
    }

    Eigen::Matrix<T, -1, 1> get_velocity() const
    {
        return v_;
    }
    Eigen::Matrix<T, -1, 1> get_position() const
    {
        return x_;
    }
    Eigen::Matrix<T, -1, 1> get_mass() const
    {
        return mass_;
    }

    Eigen::Matrix<T, -1, 1> solve(const Eigen::Matrix<T, -1, 1>& Jaccobi)
    {
        x_ = x_ + h_ * v_;
        v_ = v_ + (h_ * mass_.cwiseProduct(Jaccobi));
        return x_;
    }

private:
    Eigen::Matrix<T, -1, 1> v_;
    Eigen::Matrix<T, -1, 1> x_;
    Eigen::Matrix<T, -1, 1> mass_;
    T                       h_;
};

#endif  // SEMI_IMPLICIT_EUELER_JJ_H
