/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: constraint interface
 * @version    : 1.0
 */
#ifndef BIGBANG_CONSTRAINTS
#define BIGBANG_CONSTRAINTS
#include <unordered_map>
#include "Common/def.h"

namespace PhysIKA {
class hard_position_constraint : public Constraint<double>
{
public:
    typedef Eigen::Triplet<double> TPL;
    hard_position_constraint(const Eigen::Matrix<double, -1, -1>& nods);
    size_t Nx() const;
    size_t Nf() const;
    int    Val(const double* x, double* val) const;
    int    Jac(const double* x, const size_t off, std::vector<TPL>* jac) const;
    int    Hes(const double* x, const size_t off, std::vector<std::vector<TPL>>* hes) const;
    template <class Container>
    void update_fixed_verts(const Container& pids, const double* pos)
    {
        if (rd_ == 3)
        {
            fixed3d_.clear();
            for (auto& id : pids)
            {
                assert(id >= 0 && id < Nx() / 3);
                fixed3d_.insert(std::make_pair(id, Eigen::Vector3d(pos[3 * id + 0], pos[3 * id + 1], pos[3 * id + 2])));
            }
            return;
        }

        if (rd_ == 2)
        {
            fixed2d_.clear();
            for (auto& id : pids)
            {
                assert(id >= 0 && id < Nx() / 2);
                fixed2d_.insert(std::make_pair(id, Eigen::Vector2d(pos[2 * id + 0], pos[2 * id + 1])));
            }
            return;
        }
    }

private:
    const size_t                                dim_, rd_;
    std::unordered_map<size_t, Eigen::Vector3d> fixed3d_;
    std::unordered_map<size_t, Eigen::Vector2d> fixed2d_;
};

template <typename T>
class constraint_4_coll : public Constraint<T>
{
public:
    virtual ~constraint_4_coll()                                                                = default;
    virtual size_t Nx() const                                                                   = 0;
    virtual size_t Nf() const                                                                   = 0;
    virtual int    Val(const T* x, T* val) const                                                = 0;
    virtual int    Jac(const T* x, const size_t off, std::vector<Eigen::Triplet<T>>* jac) const = 0;
    virtual int    update(const T* x)                                                           = 0;

    // use this function to do collision detect.
    // Jac and Val just return the last collision detect infomation.
    virtual bool verify_no_collision(const T* x_new) = 0;
};

}  // namespace PhysIKA
#endif
