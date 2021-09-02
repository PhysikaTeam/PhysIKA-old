#include "../inc/mesh.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <map>
#include <set>
#include <sstream>
#include <numeric>
#include <algorithm>
#include "../inc/vert.h"
#include "../inc/write_to_file.h"
#include "predicates.h"
#include "../inc/is_vert_inside_triangle_2d.h"

using namespace std;
using namespace Eigen;

size_t Mesh::get_axis_grid(
    size_t                axis_itr,
    const vector<size_t>& sequence,
    const vector<double>& grid_axis)
{
    size_t num_v = sequence.size();
    assert(num_v > 2);
    vector<size_t> table_vert_grid_id;
    for (size_t i = 0; i < num_v; ++i)
    {
        size_t      id_v  = sequence.at(i);
        const Vert* ptr_v = get_vert_ptr(id_v);
        if (ptr_v->is_grid_vert())
        {
            size_t grid = ptr_v->get_grid_line(axis_itr);
            table_vert_grid_id.push_back(grid);
            continue;
        }
        else if (ptr_v->is_triangle_vert())
        {
            assert(id_v < get_vert_num());
            Vector3d coord_v = get_vert_ptr(id_v)->get_vert_coord();
            double   p       = coord_v(axis_itr);

            auto ptr_p = lower_bound(
                grid_axis.begin(), grid_axis.end(), p);
            if (ptr_p == grid_axis.end())
            {
                return distance(grid_axis.begin(), ptr_p);
            }
            else if (*ptr_p != p)
            {
                return distance(grid_axis.begin(), ptr_p);
            }
            else
            {
                table_vert_grid_id.push_back(distance(grid_axis.begin(), ptr_p));
                continue;
            }
        }
        else if (ptr_v->is_line_vert())
        {
            size_t id_v_grid = get_line_vert_axis_grid_id(
                axis_itr, ptr_v, grid_axis, table_vert_grid_id);

            if (id_v_grid != numeric_limits<size_t>::max())
                return id_v_grid;
            else
                continue;
        }
        else
        {
            assert(ptr_v->is_edge_vert());
            size_t id_v_grid =
                get_edge_vert_axis_grid_id(axis_itr, id_v, grid_axis, table_vert_grid_id);
            if (id_v_grid != numeric_limits<size_t>::max())
                return id_v_grid;
            else
                continue;
        }
    }

    auto ptr_min = max_element(
        table_vert_grid_id.begin(), table_vert_grid_id.end());
    return *ptr_min;
}

size_t Mesh::get_line_vert_axis_grid_id(
    size_t                axis_itr,
    const Vert*           ptr_v,
    const vector<double>& grid_axis,
    vector<size_t>&       grid_id)
{
    size_t a1     = numeric_limits<size_t>::max();
    size_t a2     = numeric_limits<size_t>::max();
    size_t g1     = numeric_limits<size_t>::max();
    size_t g2     = numeric_limits<size_t>::max();
    size_t id_tri = numeric_limits<size_t>::max();
    ptr_v->get_line_vert_info(a1, a2, g1, g2, id_tri);
    assert(a1 < 3);
    assert(a2 < 3);
    assert(g1 < grid_axis.size());
    assert(g2 < grid_axis.size());
    assert(id_tri < get_tri_num());

    if (axis_itr == a1)
    {
        grid_id.push_back(g1);
        return numeric_limits<size_t>::max();
    }
    else if (axis_itr == a2)
    {
        grid_id.push_back(g2);
        return numeric_limits<size_t>::max();
    }
    else
    {
        size_t axis_cutting = 3 - a1 - a2;
        assert(axis_cutting == axis_itr);

        double p[3];
        p[a1] = get_grid_line(a1, g1);
        p[a2] = get_grid_line(a2, g2);
        return get_line_vert_grid_id(axis_itr, p, id_tri, grid_axis, grid_id);
    }
}

size_t Mesh::get_line_vert_grid_id(
    size_t                axis_itr,
    double*               p,
    size_t                id_tri,
    const vector<double>& grid_axis,
    vector<size_t>&       grid_id)
{
    const MatrixXd& aabb_m   = get_aabb();
    double          min_aabb = aabb_m(axis_itr, 0) - 1;
    double          max_aabb = aabb_m(axis_itr, 1) + 1;

    size_t num_grid_v = grid_id.size();

    vector<Vector3d> v_tri = get_tri(id_tri);
    p[axis_itr]            = min_aabb;

    int is_above = is_vert_above_triangle(
        &v_tri[0](0), &v_tri[1](0), &v_tri[2](0), p);
    assert(is_above != 0);
    for (size_t k = 0; k < grid_axis.size(); ++k)
    {
        p[axis_itr] = grid_axis[k];
        int is_ab   = is_vert_above_triangle(
            &v_tri[0](0), &v_tri[1](0), &v_tri[2](0), p);
        if (is_ab == 0)
        {
            grid_id.push_back(k);
            return numeric_limits<size_t>::max();
        }
        else if (is_ab != is_above)
        {
            return k;
        }
    }

    if (num_grid_v == grid_id.size())
    {
        p[axis_itr] = max_aabb;
        int is_ab   = is_vert_above_triangle(
            &v_tri[0](0), &v_tri[1](0), &v_tri[2](0), p);

        assert(is_ab != 0);
        assert(is_ab != is_above);
        return grid_axis.size();
    }
    assert(false);
}

size_t Mesh::get_edge_vert_axis_grid_id(
    size_t                axis_itr,
    size_t                id_v,
    const vector<double>& grid_axis,
    vector<size_t>&       grid_id)
{
    size_t v1           = numeric_limits<size_t>::max();
    size_t v2           = numeric_limits<size_t>::max();
    size_t axis         = numeric_limits<size_t>::max();
    size_t id_grid_line = numeric_limits<size_t>::max();

    get_vert_ptr(id_v)->get_edge_vert_info(v1, v2, axis, id_grid_line);
    assert(v1 < get_vert_num());
    assert(v2 < get_vert_num());
    assert(axis < 3);
    assert(id_grid_line < grid_axis.size());

    Vector3d coord_v[2];
    coord_v[0] = get_vert_ptr(v1)->get_vert_coord();
    coord_v[1] = get_vert_ptr(v2)->get_vert_coord();

    if (axis_itr == axis)
    {
        grid_id.push_back(id_grid_line);
        return numeric_limits<size_t>::max();
    }

    size_t axis_projection = 3 - axis - axis_itr;
    assert(axis_projection < 3);
    Vector2d v_projection[2];
    for (size_t k = 0; k < 2; ++k)
    {
        for (size_t q = 0; q < 2; ++q)
        {
            size_t axis_q      = (axis_projection + q + 1) % 3;
            v_projection[k](q) = coord_v[k](axis_q);
        }
    }

    size_t id_v_grid =
        set_projection_edge_vert_axis_grid_id(
            axis_projection, id_v, axis, grid_axis, id_grid_line, grid_id, v_projection);
    return id_v_grid;
}

size_t Mesh::set_projection_edge_vert_axis_grid_id(
    size_t                axis_projection,
    size_t                id_v,
    size_t                axis,
    const vector<double>& grid_axis,
    size_t                id_grid_line,
    vector<size_t>&       grid_id,
    const Vector2d        v_projection[2])
{
    const MatrixXd& aabb_m   = get_aabb();
    size_t          axis_itr = 3 - axis_projection - axis;
    double          min_aabb = aabb_m(axis_itr, 0) - 1;
    double          max_aabb = aabb_m(axis_itr, 1) + 1;

    size_t num_grid_v = grid_id.size();
    double p[2];
    size_t itr_axis = numeric_limits<size_t>::max();
    if (axis == (axis_projection + 1) % 3)
        itr_axis = 0;
    else
        itr_axis = 1;

    p[itr_axis]     = get_grid_line(axis, id_grid_line);
    p[1 - itr_axis] = min_aabb;

    int is_positive =
        is_triangle_area_positive(&v_projection[0](0), &v_projection[1](0), p);
    assert(is_positive != 0);
    for (size_t k = 0; k < grid_axis.size(); ++k)
    {
        p[1 - itr_axis] = grid_axis.at(k);
        int is_pt =
            is_triangle_area_positive(&v_projection[0](0), &v_projection[1](0), p);
        if (is_pt == 0)
        {
            grid_id.push_back(k);
            return numeric_limits<size_t>::max();
        }
        else if (is_pt != is_positive)
        {
            return k;
        }
    }

    if (num_grid_v == grid_id.size())
    {
        p[1 - itr_axis] = max_aabb;
        int is_pt =
            is_triangle_area_positive(&v_projection[0](0), &v_projection[1](0), p);
        assert(is_pt != 0);
        assert(is_pt != is_positive);
        return grid_axis.size();
    }
    assert(false);
}
