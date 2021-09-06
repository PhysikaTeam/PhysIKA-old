#include "FEMGeometryTetCutCellGenMesh.h"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <string>
#include <iomanip>
#include "FEMGeometryTetCutCellGenVert.h"
#include "FEMGeometryTetCutCellGenRing.h"
#include "FEMGeometryTetCutCellGenEdge.h"
#include "FEMGeometryTetCutCellGenIsVertInsideTri2D.h"
#include "FEMGeometryTetCutCellGenWriteToFile.h"

using namespace std;
using namespace Eigen;

int Mesh::cut_mesh()
{
    const int all_work = 3 * num_span_;
    for (size_t axis = 0; axis < 3; ++axis)
    {
        for (size_t itr = 0; itr < num_span_ - 1; ++itr)
        {
            printf("process: %lf%\n", 100.0 * (axis * num_span_ + itr) / all_work);
            // cerr << "axis:" << axis << " " << itr << endl;
            vector<Ring> table_ring        = cut_mesh_on_axis_at_p(axis, itr);
            bool         need_set_triangle = true;
            for (auto& ring : table_ring)
            {
                ring.sort_to_closed_ring();
                ring.set_loop(need_set_triangle);
                need_set_triangle = false;
            }
            // END FOR
        }
    }

    cerr << "*********************SURFACE CUTTING************************" << endl;
    cut_surface();

    return 0;
}

vector<Ring> Mesh::cut_mesh_on_axis_at_p(size_t axis, size_t p)
{
    vector<Ring> table_ring;
    Ring         ring_upper(axis, p);
    Ring         ring_lower(axis, p);

    size_t num_tri     = table_tri_.size();
    bool   is_coplanar = false;
    for (size_t id_tri = 0; id_tri < num_tri; ++id_tri)
    {
        const Vector3st& tri        = table_tri_.at(id_tri);
        vector<size_t>   table_v_id = { tri(0), tri(1), tri(2) };
        MatrixXd         aabb       = get_aabb(table_v_id);
        double           p_grid     = grid_line_(axis, p);
        // TODO: triangle grid face coplanar
        if (aabb(axis, 1) < p_grid || aabb(axis, 0) > p_grid)
            continue;

        cut_one_tri(aabb(axis, 0), aabb(axis, 1), axis, p, p_grid, id_tri, ring_lower, ring_upper, is_coplanar);
    }

    table_ring.push_back(ring_upper);

    return table_ring;
}

int Mesh::cut_edge(size_t id_tri, Edge& edge)
{
    const Vector3st& tri        = table_tri_.at(id_tri);
    vector<size_t>   table_v_id = { tri(0), tri(1), tri(2) };
    MatrixXd         aabb       = get_aabb(table_v_id);
    size_t           num_col    = grid_line_.cols();

    for (size_t itr = 1; itr <= 2; ++itr)
    {
        size_t         axis_edge         = edge.get_axis();
        size_t         id_axis_edge_grid = edge.get_grid_line();
        size_t         axis_cut          = (axis_edge + itr) % 3;
        size_t         axis_projection   = (axis_edge + 3 - itr) % 3;
        vector<double> axis_grid_line(num_col);
        for (size_t i = 0; i < num_col; ++i)
            axis_grid_line[i] = grid_line_(axis_cut, i);
        auto ptr_start = lower_bound(
            axis_grid_line.begin(), axis_grid_line.end(), aabb(axis_cut, 0));
        size_t id_grid_start = distance(axis_grid_line.begin(), ptr_start);
        auto   ptr_end       = lower_bound(
            axis_grid_line.begin(), axis_grid_line.end(), aabb(axis_cut, 1));

        size_t id_grid_end = 0;
        if (ptr_end == axis_grid_line.end())
            id_grid_end = distance(axis_grid_line.begin(), ptr_end) - 1;
        else
            id_grid_end = distance(axis_grid_line.begin(), ptr_end);

        while (id_grid_start < id_grid_end + 1)
        {
            size_t v1_e      = numeric_limits<size_t>::max();
            size_t v2_e      = numeric_limits<size_t>::max();
            int    is_inside = is_vert_inside_triangle(
                id_tri, axis_projection, axis_edge, id_axis_edge_grid, axis_cut, id_grid_start, v1_e, v2_e);

            if (is_inside != -1)
            {
                Vert*  vert_ptr = new VertLine(id_tri, axis_edge, id_axis_edge_grid, axis_cut, id_grid_start);
                size_t id_vert  = add_vert(vert_ptr);
                vert_ptr->set_id(id_vert);

                if (is_inside == 0)
                {
                    vert_ptr->set_vert_on_triangle_edge();
                    assert(v1_e != numeric_limits<size_t>::max()
                           && v2_e != numeric_limits<size_t>::max());
                    vert_ptr->set_edge_v(v1_e, v2_e);
                }
                if (is_inside == 2)
                {
                    vert_ptr->set_vert_on_triangle_vert();
                    vert_ptr->set_vert_on_triangle_edge();
                }

                edge.add_grid_vert(axis_cut, id_vert);
            }
            ++id_grid_start;
        }
    }

    edge.sort_grid_line_vert();  // important
    edge.set_edge_on_grid();

    return 0;
}

vector<size_t> Mesh::cut_triangle_on_axis_at_p(
    size_t id_tri,
    size_t axis,
    size_t p)
{
    const Vector3st& tri    = table_tri_.at(id_tri);
    vector<Vector3d> tri_v  = get_tri(id_tri);
    double           p_grid = grid_line_(axis, p);
    vector<size_t>   table_new_vert;

    size_t id_start_vert      = 0;
    bool   is_start_vert_find = false;
    for (size_t itr = 0; itr < 3; ++itr)
    {
        const Vector3d& v           = tri_v.at(itr);
        const Vector3d& v_next      = tri_v.at((itr + 1) % 3);
        double          v_axis      = v(axis);
        double          v_next_axis = v_next(axis);
        if (v_axis >= p_grid && v_next_axis < p_grid
            || v_axis > p_grid && v_next_axis <= p_grid)
        {
            id_start_vert      = itr;
            is_start_vert_find = true;
            break;
        }
    }
    assert(is_start_vert_find);

    for (size_t i = 0; i < 3; ++i)
    {
        const size_t    itr         = (i + id_start_vert) % 3;
        const Vector3d& v           = tri_v.at(itr);
        const Vector3d& v_next      = tri_v.at((itr + 1) % 3);
        double          v_axis      = v(axis);
        double          v_next_axis = v_next(axis);
        if (v_axis < p_grid && v_next_axis <= p_grid)
            continue;
        if (v_axis < p_grid && v_next_axis > p_grid)
        {
            Vert* vert_ptr = new VertEdge(tri(itr), tri((itr + 1) % 3), axis, p);
            vert_ptr->set_grid_line(axis, p);
            size_t id_vert = add_vert(vert_ptr);
            vert_ptr->set_id(id_vert);
            table_new_vert.push_back(id_vert);
        }
        if (v_axis == p_grid)
        {
            Vert* vert_ptr = table_vert_ptr_.at(tri(itr));
            vert_ptr->set_grid_line(axis, p);
            table_new_vert.push_back(tri(itr));
        }
        if (v_axis > p_grid && v_next_axis < p_grid)
        {
            Vert* vert_ptr = new VertEdge(tri(itr), tri((itr + 1) % 3), axis, p);
            vert_ptr->set_grid_line(axis, p);
            size_t id_vert = add_vert(vert_ptr);
            vert_ptr->set_id(id_vert);
            table_new_vert.push_back(id_vert);
        }
        if (v_axis > p_grid && v_next_axis >= p_grid)
            continue;
    }

    return table_new_vert;
}

size_t Mesh::add_vert(Vert* const vert_ptr)
{
    table_vert_ptr_.push_back(vert_ptr);
    return table_vert_ptr_.size() - 1;
}

// -1: outside; 0: on edge; 1 inside; 2: on edge
int Mesh::is_vert_inside_triangle(
    size_t  id_tri,
    size_t  axis,
    size_t  axis_1,
    size_t  grid_1,
    size_t  axis_2,
    size_t  grid_2,
    size_t& v1_e,
    size_t& v2_e)
{
    vector<Vector3d> table_tri_v = get_tri(id_tri);
    Vector3st        id_tri_v    = get_tri_vert_id(id_tri);
    double           p[2];
    size_t           itr1 = (3 + axis_1 - axis) % 3 - 1;
    size_t           itr2 = (3 + axis_2 - axis) % 3 - 1;
    assert(itr1 < 2 && itr2 < 2);
    p[itr1] = grid_line_(axis_1, grid_1);
    p[itr2] = grid_line_(axis_2, grid_2);
    double tri_v[3][2];
    for (size_t itr = 0; itr < 3; ++itr)
    {
        tri_v[itr][0] = table_tri_v[itr][(axis + 1) % 3];
        tri_v[itr][1] = table_tri_v[itr][(axis + 2) % 3];
    }

    int is_inside = is_vert_inside_triangle_2d(tri_v[0], tri_v[1], tri_v[2], p);
    if (is_inside == 0)
    {
        int flag =
            is_vert_inside_triangle_2d(tri_v[0], tri_v[1], tri_v[2], p, v1_e, v2_e);
        if (flag == 2)
            return flag;

        assert(v1_e != numeric_limits<size_t>::max()
               && v2_e != numeric_limits<size_t>::max());
        v1_e = id_tri_v(v1_e);
        v2_e = id_tri_v(v2_e);
    }

    return is_inside;
}

int Mesh::add_triangle_cutted_edge(size_t id_tri, const Edge& e)
{
    Triangle& tri = table_triangle_.at(id_tri);
    tri.add_cutted_edge(e);

    return 0;
}

int Mesh::add_triangle_parallel_grid_edge(
    size_t      id_tri,
    size_t      itr,
    const Edge& e)
{
    Triangle& tri  = table_triangle_.at(id_tri);
    size_t    axis = (e.get_axis()) % 3;
    // size_t axis = (e.get_axis() + itr + 1) % 3;
    tri.add_parallel_grid_edge(axis, e);

    return 0;
}

int Mesh::cut_surface()
{
    int id_f = 1;
    for (auto& tri : table_triangle_)
    {
        tri.sort_vert();
        tri.cut();

        // string str_f = "patch_" + to_string(id_f) + ".vtk";
        // ++id_f;
        // tri.write_patch_to_file(str_f.c_str());
    }

    return 0;
}

bool Mesh::is_triangle_normal_positive(size_t axis_projection, size_t id_tri)
{
    const Vector3st& tri = table_tri_.at(id_tri);

    vector<Vector3d> v_tri(3);
    for (size_t i = 0; i < 3; ++i)
    {
        const Vert* ptr_v = get_vert_ptr(tri(i));
        v_tri[i]          = ptr_v->get_vert_coord();
    }

    vector<Vector2d> v_tri_projected(3);
    for (size_t i = 0; i < 3; ++i)
    {
        Vector2d v;
        v(0)               = v_tri[i]((axis_projection + 1) % 3);
        v(1)               = v_tri[i]((axis_projection + 2) % 3);
        v_tri_projected[i] = v;
    }

    int is_positive = is_triangle_area_positive(
        &v_tri_projected[0](0), &v_tri_projected[1](0), &v_tri_projected[2](0));
    assert(is_positive != 0);

    if (is_positive < 0)
        return false;
    else
        return true;
}

int Mesh::cut_one_tri(
    double lower_t,
    double upper_t,
    size_t axis,
    size_t p,
    double p_grid,
    size_t id_tri,
    Ring&  ring_lower,
    Ring&  ring_upper,
    bool&  is_coplanar)
{
    const Vector3st& tri = table_tri_.at(id_tri);

    if (lower_t == upper_t)
    {
        assert(lower_t == p_grid);
        set_tri_coplanar(id_tri, axis, p);
        vector<Edge> table_anti_e;
        vector<Edge> table_e;
        for (size_t i = 0; i < 3; ++i)
        {
            size_t v1 = tri(i);
            size_t v2 = tri((i + 1) % 3);
            Edge   e(id_tri, v1, v2, axis, p);
            Edge   anti_e(id_tri, v2, v1, axis, p);
            table_e.push_back(e);
            table_anti_e.push_back(anti_e);
        }

        bool is_positive = is_triangle_normal_positive(axis, id_tri);
        if (is_positive)
        {
            ring_upper.add_edge(table_anti_e);
            ring_lower.add_edge(table_e);
        }
        else
        {
            ring_upper.add_edge(table_e);  // to get inner polygon without patch

            // ring_upper.add_edge(table_anti_e);
            ring_lower.add_edge(table_e);
        }
        is_coplanar = true;
        ring_upper.add_coplanar_tri_id(id_tri);
        ring_lower.add_coplanar_tri_id(id_tri);
    }
    else
    {
        vector<size_t> table_new_vert = cut_triangle_on_axis_at_p(id_tri, axis, p);
        if (table_new_vert.size() == 1)
        {
            return 0;
        }
        else if (table_new_vert.size() != 2)
        {
            cerr << "error: unexpected new vert num" << endl;
            getchar();
        }
        else
        {
            size_t id_v1 = table_new_vert.front();
            size_t id_v2 = table_new_vert.back();
            Edge   edge(id_tri, id_v1, id_v2, axis, p);
            cut_edge(id_tri, edge);
            ring_upper.add_edge(edge);
            ring_upper.add_tri_id(id_tri);

            ring_lower.add_edge(edge);
            ring_lower.add_tri_id(id_tri);
        }
    }

    return 0;
}

int Mesh::set_tri_coplanar(size_t id, size_t axis, size_t p)
{
    Triangle& tri = table_triangle_.at(id);
    tri.set_triangle_coplanar(axis, p);
    return 0;
}

int Mesh::add_polygon_front(const Vector3st&      id_lattice,
                            const vector<size_t>& polygon)
{
    for (size_t axis = 0; axis < 3; ++axis)
        assert(id_lattice[axis] != numeric_limits<size_t>::max());

    polygon_cell_[id_lattice].push_front(polygon);
    return 0;
}

int Mesh::add_polygon(const Vector3st& id_lattice, const vector<size_t>& polygon)
{
    for (size_t axis = 0; axis < 3; ++axis)
        assert(id_lattice[axis] != numeric_limits<size_t>::max());

    polygon_cell_[id_lattice].push_back(polygon);
    return 0;
}

int Mesh::add_patch(
    const Vector3st&               id_lattice,
    const pair<PatchType, size_t>& patch)
{
    for (size_t axis = 0; axis < 3; ++axis)
        assert(id_lattice[axis] != numeric_limits<size_t>::max());

    patch_cell_[id_lattice].push_back(patch);
    return 0;
}
