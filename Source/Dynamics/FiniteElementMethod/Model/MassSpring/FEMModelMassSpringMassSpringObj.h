/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: main body for mass spring method.
 * @version    : 1.0
 */
#pragma once

#include "Common/FEMCommonDef.h"
#include <unordered_map>
#include "FEMModelMassSpringKeyComparison.h"

template <typename T>
/**
 * mass spring energy class.
 *
 */
class MassSpringObj : public PhysIKA::Functional<T, 3>
{
public:
    /**
     * @brief Construct a new MassSpringObj object
     * 
     * @param path 
     * @param stiffness 
     */
    MassSpringObj(const char* path, T stiffness);
    // function of Functional
    virtual size_t Nx() const;
    virtual int    Val(const T*                                      x,
                       std::shared_ptr<PhysIKA::dat_str_core<T, 3>>& data) const;
    virtual int    Gra(const T*                                      x,
                       std::shared_ptr<PhysIKA::dat_str_core<T, 3>>& data) const;
    virtual int    Hes(const T*                                      x,
                       std::shared_ptr<PhysIKA::dat_str_core<T, 3>>& data) const;

private:
    Eigen::Matrix<T, -1, -1>                                                                                                                 verts_;
    Eigen::Matrix<int, -1, -1>                                                                                                               cells_;
    int                                                                                                                                      var_num_;
    T                                                                                                                                        stiffness_;
    std::unordered_map<std::array<int, 2>, T, PhysIKA::UnorderedKeyHash<std::array<int, 2>>, PhysIKA::UnorderedKeyEqual<std::array<int, 2>>> edge_length_;
};

template class MassSpringObj<float>;
template class MassSpringObj<double>;
