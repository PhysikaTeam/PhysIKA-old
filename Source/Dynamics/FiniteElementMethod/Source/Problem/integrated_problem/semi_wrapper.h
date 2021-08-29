/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: semi imicilpt euler solver.
 * @version    : 1.0
 */
#ifndef SEMI_WRAPPER_JJ_H
#define SEMI_WRAPPER_JJ_H

#include "Solver/semi_implicit_euler.h"

/**
 * semi implicit method wrapper.
 *
 */
template <typename T>
class semi_wrapper
{
public:
    semi_wrapper()
        : semi_implicit_(nullptr) {}
    virtual std::shared_ptr<semi_implicit<T>> get_semi_implicit() const
    {
        return semi_implicit_;
    }

protected:
    std::shared_ptr<semi_implicit<T>> semi_implicit_;
};

#endif  // SEMI_WRAPPER_JJ_H
