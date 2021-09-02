/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: main body for mass spring method.
 * @version    : 1.0
 */
#include "mass_spring_obj.h"
#include "edge_gradient_and_hessian.h"
#include "read_model.h"
#include <iostream>

using namespace std;
using namespace Eigen;
using namespace PhysIKA;

template <typename T>
MassSpringObj<T>::MassSpringObj(const char* path, T stiffness)
    : stiffness_(stiffness)
{
    bool read_success = false;
    if (string(path).find(".obj") != string::npos)
        tie(read_success, verts_, cells_) = ReadObj<T, int>(path);
    else
        tie(read_success, verts_, cells_) = ReadTetHex<4, T, int>(path);

    const int f_num = cells_.rows();
    for (int f = 0; f < f_num; ++f)
    {
        vector<int> edge_vert_idx;
        if (string(path).find(".obj") != string::npos)
            edge_vert_idx = { 0, 1, 2, 0 };
        else
            edge_vert_idx = { 0, 1, 2, 0, 3, 1, 2, 3 };

        for (int e = 0; e < edge_vert_idx.size() - 1; ++e)
        {
            array<int, 2> edge = { cells_(f, edge_vert_idx[e]), cells_(f, edge_vert_idx[e + 1]) };
            if (edge_length_.count(edge))
                continue;
            T l0 = (verts_.row(edge[0]) - verts_.row(edge[1])).norm();
            edge_length_.emplace(edge, l0);
        }
    }
    var_num_ = verts_.size();
}

template <typename T>
size_t MassSpringObj<T>::Nx() const
{
    return verts_.size();
}

template <typename T>
int MassSpringObj<T>::Val(const T*                                      x,
                          std::shared_ptr<PhysIKA::dat_str_core<T, 3>>& data) const
{
    T value = 0;
    for (const auto& e : edge_length_)
    {
        int v1   = e.first[0];
        int v2   = e.first[1];
        T   l0   = e.second;
        T   v[6] = { x[3 * v1], x[3 * v1 + 1], x[3 * v1 + 2], x[3 * v2], x[3 * v2 + 1], x[3 * v2 + 2] };
        value += GetEdgeEnergy<T>(v, stiffness_, l0);
    }

    data->save_val(value);
    return 0;
}

template <typename T>
int MassSpringObj<T>::Gra(const T*                                      x,
                          std::shared_ptr<PhysIKA::dat_str_core<T, 3>>& data) const
{
    Matrix<T, -1, -1> G = Matrix<T, -1, -1>::Zero(var_num_, 1);
    for (const auto& e : edge_length_)
    {
        int             v1   = e.first[0];
        int             v2   = e.first[1];
        T               l0   = e.second;
        T               v[6] = { x[3 * v1], x[3 * v1 + 1], x[3 * v1 + 2], x[3 * v2], x[3 * v2 + 1], x[3 * v2 + 2] };
        Matrix<T, 6, 1> g    = GetEdgeGradient<T>(v, stiffness_, l0);
        G.block(3 * v1, 0, 3, 1) += g.block(0, 0, 3, 1);
        G.block(3 * v2, 0, 3, 1) += g.block(3, 0, 3, 1);
    }

    data->save_gra(G);
    return 0;
}

template <typename T>
int MassSpringObj<T>::Hes(const T*                                      x,
                          std::shared_ptr<PhysIKA::dat_str_core<T, 3>>& data) const
{
    for (const auto& e : edge_length_)
    {
        int             v1     = e.first[0];
        int             v2     = e.first[1];
        T               l0     = e.second;
        T               v[6]   = { x[3 * v1], x[3 * v1 + 1], x[3 * v1 + 2], x[3 * v2], x[3 * v2 + 1], x[3 * v2 + 2] };
        Matrix<T, 6, 6> H      = GetEdgeHessian<T>(v, stiffness_, l0);
        int             idx[6] = { 3 * v1, 3 * v1 + 1, 3 * v1 + 2, 3 * v2, 3 * v2 + 1, 3 * v2 + 2 };
        for (int i = 0; i < 6; ++i)
        {
            for (int j = 0; j < 6; ++j)
            {
                data->save_hes(idx[i], idx[j], H(i, j));
            }
        }
    }

    return 0;
}
