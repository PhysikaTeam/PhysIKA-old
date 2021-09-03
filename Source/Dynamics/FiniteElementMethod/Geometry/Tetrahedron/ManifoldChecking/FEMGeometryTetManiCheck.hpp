#pragma once

#include "FEMGeometryTetManiCheck.h"
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <vector>
#include <numeric>
#include <Eigen/Core>
#include <random>
#include "FEMGeometryTetManiCheckKDTree.hpp"
#include "Predicates.h"

using namespace std;
using namespace Eigen;

template <typename T>
bool is_mesh_self_intersection(const char* const path)
{
    KdTree<T> kd_tree;
    kd_tree.build_tree(path);

    size_t num_tri = kd_tree.get_tri_num();
    for (size_t i = 0; i < num_tri; ++i)
    {
        const Matrix<T, 3, 3>   tri       = kd_tree.get_tri(i);
        vector<Matrix<T, 3, 3>> neigh_tri = kd_tree.get_neigh_tri(i);

        for (const auto& itr : neigh_tri)
        {
            bool is_intersect = is_triangle_triangle_self_intersect(tri, itr);
            if (is_intersect)
            {
                cerr << "self intersect occur" << endl;
                return true;
            }
        }
    }

    return false;
}

template <typename T>
bool is_triangle_triangle_self_intersect(const Eigen::Matrix<T, 3, 3>& tri_a, const Eigen::Matrix<T, 3, 3>& tri_b)
{
    array<int, 3> position_a2b;
    array<int, 3> position_b2a;
    for (size_t i = 0; i < 3; ++i)
    {
        double a2b      = orient3d(&tri_b(0, 0), &tri_b(0, 1), &tri_b(0, 2), &tri_a(0, i));
        double b2a      = orient3d(&tri_a(0, 0), &tri_a(0, 1), &tri_a(0, 2), &tri_b(0, i));
        position_a2b[i] = (a2b == 0 ? 0 : a2b > 0 ? 1
                                                  : -1);
        position_b2a[i] = (b2a == 0 ? 0 : b2a > 0 ? 1
                                                  : -1);
    }

    bool is_a_degenerate = is_triangle_degenerate<T>(tri_a);
    bool is_b_degenerate = is_triangle_degenerate<T>(tri_b);
    if (is_a_degenerate || is_b_degenerate)
        return true;
    int is_common = is_common_condition<T>(tri_a, tri_b, position_a2b, position_b2a);
    if (is_common == 0)
        return false;
    else if (is_common == 1)
        return true;
    bool is_it = is_triangle_triangle_self_intersect(tri_a, tri_b, position_a2b, position_b2a);
    return is_it;
}

bool is_triangle_above_triangle(const std::array<int, 3>& position)
{
    for (size_t i = 0; i < 3; ++i)
    {
        if (position[i] * position[(i + 1) % 3] <= 0)
            return false;
    }

    return true;
}

template <typename T>
int is_common_condition(const Eigen::Matrix<T, 3, 3>& tri_a, const Eigen::Matrix<T, 3, 3>& tri_b, const std::array<int, 3>& position_a2b, const std::array<int, 3>& position_b2a)
{
    bool is_a_above_b = is_triangle_above_triangle(position_a2b);
    bool is_b_above_a = is_triangle_above_triangle(position_b2a);
    if (is_a_above_b || is_b_above_a)
        return 0;
    array<int, 3> idx = { 0, 1, 2 };
    int           num_coincidence =
        accumulate(idx.begin(), idx.end(), 0, [&tri_a, &tri_b](int b, int v) -> int {
            return b + (tri_b.col(v) == tri_a.col(0))
                   + (tri_b.col(v) == tri_a.col(1))
                   + (tri_b.col(v) == tri_a.col(2));
        });
    if (num_coincidence == 1)
    {
        bool is_a_universal = is_universal_connect(position_a2b);
        bool is_b_universal = is_universal_connect(position_b2a);
        if (is_a_universal || is_b_universal)
            return 0;
    }
    else if (num_coincidence == 2)
    {
        int sum_coplanar = accumulate(position_a2b.begin(), position_a2b.end(), 0, [](int b, int v) -> int {
            return b + (v == 0);
        });
        if (sum_coplanar == 2)
            return 0;
    }
    else if (num_coincidence == 3)
    {
        return 1;
    }

    return 2;
}

bool is_universal_connect(const std::array<int, 3>& position)
{
    int sum_coplanar = accumulate(position.begin(), position.end(), 0, [](int b, int v) -> int {
        return b + (v == 0);
    });
    if (sum_coplanar == 1)
    {
        auto ptr = find(position.begin(), position.end(), 0);
        int  idx = distance(position.begin(), ptr);
        if (position.at((idx + 1) % 3) * position.at((idx + 2) % 3) > 0)
            return true;
    }

    return false;
}

template <typename T>
bool is_triangle_degenerate(const Eigen::Matrix<T, 3, 3>& tri)
{
    for (size_t i = 0; i < 3; ++i)
    {
        if (tri.col(i) == tri.col((i + 1) % 3))
            return true;
    }

    vector<Matrix<T, 3, 1>> ot = { Matrix<T, 3, 1>(1, 0, 0),
                                   Matrix<T, 3, 1>(0, 1, 0),
                                   Matrix<T, 3, 1>(0, 0, 1),
                                   Matrix<T, 3, 1>(-1, 0, 0) };
    for (size_t i = 0; i < 4; ++i)
    {
        if (orient3d(&tri(0, 0), &tri(0, 1), &tri(0, 2), &ot[i][0]) != 0)
            return false;
    }

    return true;
}

template <typename T>
bool is_triangle_triangle_self_intersect(const Eigen::Matrix<T, 3, 3>& tri_a, const Eigen::Matrix<T, 3, 3>& tri_b, const std::array<int, 3>& position_a2b, const std::array<int, 3>& position_b2a)
{
    int sum_coplanar = accumulate(position_a2b.begin(), position_a2b.end(), 0, [](int b, int v) -> int {
        return b + (v == 0);
    });
    if (sum_coplanar == 3)
    {
        int sum_dual_coplanar = accumulate(position_b2a.begin(), position_b2a.end(), 0, [](int b, int v) -> int {
            return b + (v == 0);
        });
        assert(sum_dual_coplanar == 3);
        bool is_interset = is_coplanar_triangle_triangle_intersect(tri_a, tri_b);
        return is_interset;
    }
    else
    {
        bool is_a2b_intersect = is_triangle_to_triangle_intersect(tri_a, tri_b);
        bool is_b2a_intersect = is_triangle_to_triangle_intersect(tri_b, tri_a);
        if (is_a2b_intersect || is_b2a_intersect)
            return true;
        else
            return false;
    }
}

template <typename T>
bool is_coplanar_triangle_triangle_intersect(const Eigen::Matrix<T, 3, 3>& tri_a, const Eigen::Matrix<T, 3, 3>& tri_b)
{
    unsigned                     seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine        e(seed1);
    uniform_real_distribution<T> u(1, 2);
    Matrix<T, 3, 1>              disturb(u(e), u(e), u(e));
    Matrix<T, 3, 1>              p = tri_a.col(1) + disturb;
    while (true)
    {
        double dis_pa = orient3d(&tri_a(0, 0), &tri_a(0, 1), &tri_a(0, 2), &p[0]);
        double dis_pb = orient3d(&tri_b(0, 0), &tri_b(0, 1), &tri_b(0, 2), &p[0]);
        if (dis_pa * dis_pb == 0)
        {
            Matrix<T, 3, 1> disturb(u(e), u(e), u(e));
            p = tri_a.col(1) + disturb;
        }
        else
            break;
    }

    array<Matrix<T, 3, 3>, 2> tri = { tri_a, tri_b };
    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = 0; j < 2; ++j)
        {
            const Matrix<T, 3, 1>& v            = tri[j].col(i);
            bool                   is_intersect = is_vert_triangle_intersect(v, p, tri[1 - j]);
            if (is_intersect)
                return true;
        }
    }

    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = 0; j < 2; ++j)
        {
            Matrix<T, 3, 2> e;
            e.col(0)          = tri[j].col(i);
            e.col(1)          = tri[j].col((i + 1) % 3);
            bool is_intersect = is_coplanar_edge_triangle_intersect(e, p, tri[1 - j]);
            if (is_intersect)
                return true;
        }
    }

    return false;
}

template <typename T>
bool is_vert_triangle_intersect(const Eigen::Matrix<T, 3, 1>& v, const Eigen::Matrix<T, 3, 1>& p, const Eigen::Matrix<T, 3, 3>& tri, bool coplanar_v_triangle)
{
    assert(p != v);
    array<int, 3> position;
    for (size_t i = 0; i < 3; ++i)
    {
        double dis  = orient3d(&p[0], &tri(0, i), &tri(0, (i + 1) % 3), &v[0]);
        position[i] = (dis == 0 ? 0 : dis > 0 ? 1
                                              : -1);
    }

    int sum_tet_plane = accumulate(position.begin(), position.end(), 0, [](int b, int p) -> int {
        return b + (p == 0);
    });
    if (sum_tet_plane == 0)
    {
        bool is_inner = (position[0] * position[1] > 0)
                        && (position[1] * position[2] > 0)
                        && (position[2] * position[0] > 0);
        if (is_inner)
        {
            if (coplanar_v_triangle)
                return true;
            else
            {
                double dis_v = orient3d(&tri(0, 0), &tri(0, 1), &tri(0, 2), &v(0));
                double dis_p = orient3d(&tri(0, 0), &tri(0, 1), &tri(0, 2), &p(0));
                if (dis_v * dis_p > 0)
                    return false;
                else
                    return true;
            }
        }
        else
            return false;
    }
    else if (sum_tet_plane == 1)
    {
        auto   ptr = find(position.begin(), position.end(), 0);
        size_t idx = distance(position.begin(), ptr);
        if (position[(idx + 1) % 3] * position[(idx + 2) % 3] > 0)
        {
            if (coplanar_v_triangle)
                return true;
            else
            {
                double dis_v = orient3d(&tri(0, 0), &tri(0, 1), &tri(0, 2), &v(0));
                double dis_p = orient3d(&tri(0, 0), &tri(0, 1), &tri(0, 2), &p(0));
                if (dis_v * dis_p > 0)
                    return false;
                else
                    return true;
            }
        }
        else
            return false;
    }
    else if (sum_tet_plane == 2)
    {
        if (coplanar_v_triangle)
            return false;
        else
        {
            double dis_v = orient3d(&tri(0, 0), &tri(0, 1), &tri(0, 2), &v(0));
            double dis_p = orient3d(&tri(0, 0), &tri(0, 1), &tri(0, 2), &p(0));
            if (dis_v * dis_p > 0)
                return false;
            else
                return true;
        }
    }
    else
    {
        assert(false);
    }
}

template <typename T>
bool is_coplanar_edge_triangle_intersect(const Eigen::Matrix<T, 3, 2>& e, const Eigen::Matrix<T, 3, 1>& p, const Eigen::Matrix<T, 3, 3>& tri)
{
    for (size_t i = 0; i < 3; ++i)
    {
        Matrix<T, 3, 2> et;
        et.col(0)           = tri.col(i);
        et.col(1)           = tri.col((i + 1) % 3);
        bool is_e_intersect = is_edge_edge_intersect(e, et, p);
        if (is_e_intersect)
            return true;
    }

    return false;
}

template <typename T>
bool is_edge_edge_intersect(const Eigen::Matrix<T, 3, 2>& e, const Eigen::Matrix<T, 3, 2>& et, const Eigen::Matrix<T, 3, 1>& p)
{
    array<double, 2> e2et;
    array<double, 2> et2e;
    for (size_t i = 0; i < 2; ++i)
    {
        et2e[i] = orient3d(&p[0], &e(0, 0), &e(0, 1), &et(0, i));
        e2et[i] = orient3d(&p[0], &et(0, 0), &et(0, 1), &e(0, i));
    }

    if (et2e[0] * et2e[1] >= 0)
        return false;
    if (e2et[0] * e2et[1] >= 0)
        return false;

    return true;
}

template <typename T>
bool is_triangle_to_triangle_intersect(const Eigen::Matrix<T, 3, 3>& tri_a, const Eigen::Matrix<T, 3, 3>& tri_b)
{
    for (size_t i = 0; i < 3; ++i)
    {
        Matrix<T, 3, 2> e;
        e.col(0)          = tri_a.col(i);
        e.col(1)          = tri_a.col((i + 1) % 3);
        bool is_intersect = is_edge_triangle_intersect(e, tri_b);
        if (is_intersect)
            return true;
    }

    return false;
}

template <typename T>
bool is_edge_triangle_intersect(const Eigen::Matrix<T, 3, 2>& e, const Eigen::Matrix<T, 3, 3>& tri)
{
    array<int, 2> position;
    for (size_t i = 0; i < 2; ++i)
    {
        double dis  = orient3d(&tri(0, 0), &tri(0, 1), &tri(0, 2), &e(0, i));
        position[i] = (dis == 0 ? 0 : dis > 0 ? 1
                                              : -1);
    }

    int sum_coplanar = accumulate(position.begin(), position.end(), 0, [](int b, int p) -> int {
        return b + (p == 0);
    });
    if (sum_coplanar == 2)
    {
        unsigned                     seed1 = std::chrono::system_clock::now().time_since_epoch().count();
        default_random_engine        eg(seed1);
        uniform_real_distribution<T> u(1, 2);
        Matrix<T, 3, 1>              disturb(u(eg), u(eg), u(eg));
        Matrix<T, 3, 1>              p = tri.col(0) + disturb;
        while (true)
        {
            double dis = orient3d(&tri(0, 0), &tri(0, 1), &tri(0, 2), &p[0]);
            if (dis == 0)
            {
                Matrix<T, 3, 1> disturb(u(eg), u(eg), u(eg));
                p = tri.col(0) + disturb;
            }
            else
                break;
        }
        const Matrix<T, 3, 1>& ea              = e.col(0);
        const Matrix<T, 3, 1>& eb              = e.col(1);
        bool                   is_ea_intersect = is_vert_triangle_intersect(ea, p, tri);
        bool                   is_eb_intersect = is_vert_triangle_intersect(eb, p, tri);
        bool                   is_intersect    = is_coplanar_edge_triangle_intersect(e, p, tri);
        if (is_ea_intersect || is_eb_intersect || is_intersect)
            return true;
    }
    else if (sum_coplanar == 1)
    {
        auto                   ptr          = find(position.begin(), position.end(), 0);
        size_t                 idx          = distance(position.begin(), ptr);
        const Matrix<T, 3, 1>& v            = e.col(idx);
        const Matrix<T, 3, 1>& p            = e.col(1 - idx);
        bool                   is_intersect = is_vert_triangle_intersect(v, p, tri);
        if (is_intersect)
            return true;
    }
    else
    {
        const Matrix<T, 3, 1>& v            = e.col(0);
        const Matrix<T, 3, 1>& p            = e.col(1);
        bool                   is_intersect = is_vert_triangle_intersect(v, p, tri, false);
        if (is_intersect)
            return true;
    }

    return false;
}
