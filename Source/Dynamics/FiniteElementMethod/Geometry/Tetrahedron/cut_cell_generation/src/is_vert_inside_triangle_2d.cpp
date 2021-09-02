#include "../inc/is_vert_inside_triangle_2d.h"

#include <cstddef>

using namespace std;
using namespace Eigen;

// 1: inside; 0: on edge; -1: outside
int is_vert_inside_triangle_2d(const double* const a, const double* const b, const double* const c, const double* const p)
{
    double tri_area = orient2d(a, b, c);
    if (tri_area == 0)
        return -1;

    double p_edge_relation[3];
    p_edge_relation[0] = orient2d(a, b, p);
    p_edge_relation[1] = orient2d(b, c, p);
    p_edge_relation[2] = orient2d(c, a, p);

    int is_outside = false;
    int is_on_edge = false;
    for (size_t itr = 0; itr < 3; ++itr)
    {
        if (p_edge_relation[itr] * p_edge_relation[(itr + 1) % 3] < 0)
            is_outside = true;
        if (p_edge_relation[itr] == 0)
            is_on_edge = true;
    }

    if (is_outside == false && is_on_edge == true)
        return 0;
    if (is_outside)
        return -1;
    else
        return 1;
}

int is_vert_inside_triangle_2d(const double* const a, const double* const b, const double* const c, const double* const p, size_t& v1_e, size_t& v2_e)
{
    assert(is_vert_inside_triangle_2d(a, b, c, p) == 0);
    double p_edge_relation[3];
    p_edge_relation[0] = orient2d(a, b, p);
    p_edge_relation[1] = orient2d(b, c, p);
    p_edge_relation[2] = orient2d(c, a, p);

    size_t sum_zero = 0;
    for (size_t v = 0; v < 3; ++v)
    {
        if (p_edge_relation[v] == 0)
            sum_zero += 1;
    }

    if (sum_zero == 2)
        return 2;

    assert(sum_zero == 1);
    for (size_t itr = 0; itr < 3; ++itr)
    {
        if (p_edge_relation[itr] == 0)
        {
            v1_e = itr;
            v2_e = (itr + 1) % 3;
            break;
        }
    }

    return 0;
}

int is_vert_above_triangle(const double* const a, const double* const b, const double* const c, const double* const p)
{
    double pos = orient3d(a, b, c, p);
    if (pos < 0)
        return 1;
    else if (pos > 0)
        return -1;
    else
        return 0;
}

int is_triangle_area_positive(
    size_t                              axis,
    const std::vector<Eigen::Vector3d>& tri_v)
{
    vector<double> tri_v_2d[3];
    for (size_t id_v = 0; id_v < 3; ++id_v)
    {
        for (size_t d = 0; d < 2; ++d)
        {
            size_t d_axis = (axis + d + 1) % 3;
            tri_v_2d[id_v].push_back(tri_v[id_v](d_axis));
        }
    }

    return is_triangle_area_positive(
        &tri_v_2d[0][0], &tri_v_2d[1][0], &tri_v_2d[2][0]);
}

int is_triangle_area_positive(
    double const* pa,
    double const* pb,
    double const* pc)
{
    double area = orient2d(pa, pb, pc);
    if (area > 0)
        return 1;
    else if (area < 0)
        return -1;
    else
        return 0;
}
