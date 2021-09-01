/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: functional and constraint definition
 * @version    : 1.0
 */
#ifndef NUMERIC_DEF_H
#define NUMERIC_DEF_H

#include <memory>
#include <iostream>
#include <Eigen/Sparse>

#include "data_str_core.h"
#include "error.h"
#include "config.h"
namespace PhysIKA {

template <typename T, size_t dim>
using data_ptr = std::shared_ptr<dat_str_core<T, dim>>;

/**
 * Functional interface.
 *
 * sample usage:
 * Functional->Val(x, data); // to get value.
 * Functional->Gra(x, data); // to get gradient.
 * Functional->Hes(x, data); // to get hessian.
 *
 */
template <typename T, size_t dim>
class Functional
{
public:
    virtual ~Functional() {}
    virtual size_t Nx() const                                    = 0;
    virtual int    Val(const T* x, data_ptr<T, dim>& data) const = 0;
    virtual int    Gra(const T* x, data_ptr<T, dim>& data) const = 0;
    virtual int    Hes(const T* x, data_ptr<T, dim>& data) const = 0;
    virtual int    Val_Gra_Hes(const T* x, data_ptr<T, dim>& data) const
    {
        IF_ERR(return, Val(x, data));
        IF_ERR(return, Gra(x, data));
        IF_ERR(return, Hes(x, data));
        return 0;
    }
};

/**
 * Constraint
 *
 * sample usage:
 * Functional->Val(x, data); // to get value.
 * Functional->Jac(x, data); // to get Jacobian
 * Functional->Hes(x, data); // to get hessian.
 *
 */
template <typename T>
class Constraint
{
public:
    virtual ~Constraint() {}
    virtual size_t Nx() const                                                                   = 0;
    virtual size_t Nf() const                                                                   = 0;
    virtual int    Val(const T* x, T* val) const                                                = 0;
    virtual int    Jac(const T* x, const size_t off, std::vector<Eigen::Triplet<T>>* jac) const = 0;
    virtual int    Hes(const T* x, const size_t off, std::vector<std::vector<Eigen::Triplet<T>>>* hes) const
    {
        return __LINE__;
    }
};

/**
 * null input exception, if no input, then throw this exception.
 *
 */
class null_input_exception : public std::exception
{
public:
    const char* what() const throw()
    {
        return "null input exception";
    }
};
/**
 * compatibility exception, if not compatible, then throw this exception.
 *
 */
class compatibility_exception : public std::exception
{
public:
    const char* what() const throw()
    {
        return "compatibility exception";
    }
};
template <typename T, size_t dim>
class energy_t;

template <typename T, size_t dim>
std::shared_ptr<energy_t<T, dim>> build_energy_t(const std::vector<std::shared_ptr<Functional<T, dim>>>& buffer)
{
    size_t total_dim = -1;
    for (auto& e : buffer)
    {
        if (e.get())
        {
            total_dim = e->Nx();
            break;
        }
    }
    if (total_dim == -1)
    {
        throw null_input_exception();
    }
    for (auto& e : buffer)
    {
        if (e.get() && e->Nx() != total_dim)
        {
            throw compatibility_exception();
        }
    }
    return std::make_shared<energy_t<T, dim>>(buffer, total_dim);
}

/**
 * energy class. the collection for some functionals.
 *
 */
template <typename T, size_t dim>
class energy_t : public Functional<T, dim>
{
public:
    energy_t(const std::vector<std::shared_ptr<Functional<T, dim>>>& buffer, const size_t total_dim)
        : buffer_(buffer), dim_(total_dim) {}

public:
    size_t Nx() const override
    {
        return dim_;
    }
    int Val(const T* x, std::shared_ptr<dat_str_core<T, dim>>& data) const
    {
        assert(x);
        for (auto& e : buffer_)
        {
            if (e.get())
            {
                IF_ERR(return, e->Val(x, data));
            }
        }
        return 0;
    }
    int Gra(const T* x, std::shared_ptr<dat_str_core<T, dim>>& data) const
    {
        assert(x);
        for (auto& e : buffer_)
        {
            if (e.get())
            {
                IF_ERR(return, e->Gra(x, data));
            }
        }
        return 0;
    }
    int Hes(const T* x, std::shared_ptr<dat_str_core<T, dim>>& data) const
    {
        assert(x);
        for (auto& e : buffer_)
        {
            if (e.get())
            {
                IF_ERR(return, e->Hes(x, data));
            }
        }
        return 0;
    }

public:
    const std::vector<std::shared_ptr<Functional<T, dim>>>& buffer_;
    size_t                                                  dim_;
};

template <typename T>
class constraint_t;

template <typename T>
std::shared_ptr<constraint_t<T>> build_constraint_t(const std::vector<std::shared_ptr<Constraint<T>>>& buffer)
{
    size_t xdim = -1;
    for (auto& e : buffer)
    {
        if (e.get())
        {
            xdim = e->Nx();
            break;
        }
    }
    if (xdim == -1)
        throw null_input_exception();

    bool compatible = true;
    for (auto& c : buffer)
    {
        if (c.get())
        {
            if (c->Nx() != xdim)
                compatible = false;
        }
    }
    if (!compatible)
        throw compatibility_exception();

    return std::make_shared<constraint_t<T>>(buffer, xdim);
}

/**
 * constraint type class, collection of some constraint.
 *
 */
template <typename T>
class constraint_t : public Constraint<T>
{

public:
    template <typename T2>
    friend std::shared_ptr<constraint_t<T2>> build_constraint_t(const std::vector<std::shared_ptr<Constraint<T2>>>& buffer);

    constraint_t(const std::vector<std::shared_ptr<Constraint<T>>>& buffer, const size_t xdim)
        : buffer_(buffer), xdim_(xdim) {}

public:
    size_t Nx() const
    {
        return xdim_;
    }
    size_t Nf() const
    {
        size_t fdim = 0;
        for (auto& c : buffer_)
        {
            if (c.get())
                fdim += c->Nf();
        }
        return fdim;
    }
    int Val(const T* x, T* val) const
    {
        assert(x && val);
        Eigen::Map<Eigen::Matrix<T, -1, 1>> v(val, Nf());
        size_t                              offset = 0;
        for (auto& c : buffer_)
        {
            if (c.get())
            {
                const size_t            nf = c->Nf();
                Eigen::Matrix<T, -1, 1> value(nf);
                value.setZero();
                IF_ERR(return, c->Val(x, value.data()));
                v.segment(offset, nf) += value;
                offset += nf;
            }
        }
        return 0;
    }
    int Jac(const T* x, const size_t off, std::vector<Eigen::Triplet<T>>* jac) const
    {
        assert(x && jac);
        size_t offset = off;
        for (auto& c : buffer_)
        {
            if (c.get())
            {
                IF_ERR(return, c->Jac(x, offset, jac));
                offset += c->Nf();
            }
        }
        return 0;
    }
    int Hes(const T* x, const size_t off, std::vector<std::vector<Eigen::Triplet<T>>>* hes) const
    {
        assert(x && hes);
        const size_t fdim = Nf();
        if (hes->size() != fdim)
            hes->resize(fdim);
        size_t offset = 0;
        for (auto& c : buffer_)
        {
            if (c.get())
            {
                IF_ERR(return, c->Hes(x, offset, hes));
                offset += c->Nf();
            }
        }
        return 0;
    }
    int update(const T* x)
    {
        assert(x);
        for (auto& c : buffer_)
        {
            if (c.get())
            {
                IF_ERR(return, c->update(x));
            }
        }
        return 0;
    }

protected:
    const std::vector<std::shared_ptr<Constraint<T>>>& buffer_;
    size_t                                             xdim_;
};

template <typename T, size_t field>
int compute_hes_pattern(const std::shared_ptr<Functional<T, field>>& energy,
                        std::shared_ptr<dat_str_core<T, field>>&     dat_str)
{
    const size_t total_dim = energy->Nx();
    dat_str->set_zero();
    Eigen::Matrix<T, -1, 1> random_x(total_dim);
    {
#pragma omp parallel for
        for (size_t i = 0; i < total_dim; ++i)
        {
            random_x(i) = i * 4.5 + i * i;
        }
        dat_str->set_zero();
        __TIME_BEGIN__;
        IF_ERR(return, energy->Hes(random_x.data(), dat_str));
        dat_str->setFromTriplets();
        const auto sm1 = dat_str->get_hes();
        std::cout << "the number of nonzeros with comparison: \n"
                  << (Eigen::Map<const Eigen::Matrix<T, -1, 1>>(sm1.valuePtr(), sm1.nonZeros()).array() != 0).count()
                  << std::endl;
        std::cout << "sparcity: " << T(sm1.nonZeros()) / T((sm1.rows() * sm1.cols())) << std::endl;
        dat_str->set_hes_zero_after_pre_compute();
        __TIME_END__("[INFO] Pre_compute_hes");
        return 0;
    }
}

}  // namespace PhysIKA

#endif  // NUMERIC_DEF_H
