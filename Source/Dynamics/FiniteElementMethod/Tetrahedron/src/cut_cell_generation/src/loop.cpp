#include "../inc/ring.h"
#include <Eigen/Core>
#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stack>
#include "../inc/mesh.h"
#include "../inc/edge.h"
#include "../inc/write_to_file.h"
#include "../inc/is_vert_inside_triangle_2d.h"

using namespace std;
using namespace Eigen;

Loop::Loop(Ring* ring)
{
    ring_ = ring;
}

int Loop::set_grid_line_id()
{
    vector<size_t> table_tri_id;
    for (const auto& one_ring_tri : ring_->table_ring_tri_id_)
    {
        table_tri_id.insert(
            table_tri_id.end(), one_ring_tri.begin(), one_ring_tri.end());
    }

    vector<size_t> table_vert_id;
    for (auto itr : table_tri_id)
    {
        const Vector3st& id_tri_v = ring_->mesh_->get_tri_vert_id(itr);
        for (size_t id_v = 0; id_v < 3; ++id_v)
            table_vert_id.push_back(id_tri_v(id_v));
    }
    aabb_ = ring_->mesh_->get_aabb(table_vert_id);

    const MatrixXd& grid_line = ring_->mesh_->get_grid_line();
    const size_t    num_col   = grid_line.cols();
    for (size_t itr = 0; itr < 2; ++itr)
    {
        size_t axis = (ring_->axis_ + itr + 1) % 3;

        vector<double> axis_grid_line(num_col);
        for (size_t i = 0; i < num_col; ++i)
            axis_grid_line[i] = grid_line(axis, i);

        auto ptr_lower = lower_bound(
            axis_grid_line.begin(), axis_grid_line.end(), aabb_(axis, 0));
        auto ptr_upper = lower_bound(
            axis_grid_line.begin(), axis_grid_line.end(), aabb_(axis, 1));

        size_t id_lower = distance(axis_grid_line.begin(), ptr_lower);
        size_t id_upper = id_lower;
        if (ptr_upper == axis_grid_line.end())
            id_upper = distance(axis_grid_line.begin(), ptr_upper) - 1;
        else
            id_upper = distance(axis_grid_line.begin(), ptr_upper);

        while (id_lower <= id_upper)
        {
            table_grid_line_id_[itr].push_back(id_lower);
            ++id_lower;
        }
    }

    return 0;
}

// TODO: edge parallel to grid not considered, finished
int Loop::set_vert_on_one_line()
{
    for (const auto& ring : ring_->table_ring_)
    {
        auto vert_line = vert_one_line_;
        auto edge_line = edge_one_line_;
        vert_one_line_[0].clear();
        vert_one_line_[1].clear();
        edge_one_line_[0].clear();
        edge_one_line_[1].clear();

        list<Edge> table_e;
        table_e.insert(table_e.end(), ring.begin(), ring.end());

        for (const auto& e : table_e)
        {
            assert(e.get_id() != numeric_limits<size_t>::max());
            set_edge_parallel(e);
        }

        set_invalid_edge();
        if (vert_invalid_one_line_[0].size() + vert_invalid_one_line_[1].size() != 0)
            cerr << "invalid vert find" << endl;

        size_t axis = ring_->axis_;
        for (const auto& e : table_e)
        {
            size_t grid_1 = numeric_limits<size_t>::max();
            size_t grid_2 = numeric_limits<size_t>::max();

            const vector<size_t>& table_vert_id = e.get_line_vert_id();

            for (auto id_v : table_vert_id)
            {
                const Vert* ptr_vert  = ring_->mesh_->get_vert_ptr(id_v);
                size_t      id_grid_1 = ptr_vert->get_grid_line((axis + 1) % 3);
                size_t      id_grid_2 = ptr_vert->get_grid_line((axis + 2) % 3);
                if (id_grid_1 != numeric_limits<size_t>::max())
                {
                    assert(id_grid_2 == numeric_limits<size_t>::max());
                    if (vert_invalid_one_line_[0][id_grid_1].count(id_v))
                    {
                        cerr << "AAAAAAAAAAAAAAAAAAAAAAAAAAAAa" << endl;
                        continue;
                    }

                    vert_one_line_[0][id_grid_1].push_back(id_v);
                    edge_one_line_[0][id_grid_1].push_back(e);
                }
                else
                {
                    assert(id_grid_2 != numeric_limits<size_t>::max());
                    if (vert_invalid_one_line_[1][id_grid_2].count(id_v))
                    {
                        cerr << "BBBBBBBBBBBBBBBBBBBBBBBBB" << endl;
                        continue;
                    }

                    vert_one_line_[1][id_grid_2].push_back(id_v);
                    edge_one_line_[1][id_grid_2].push_back(e);
                }
            }
        }
        remove_duplicate_vert_on_one_line();
        for (size_t i = 0; i < 2; ++i)
        {
            for (const auto& line : vert_line[i])
            {
                size_t                grid   = line.first;
                const vector<size_t>& line_v = line.second;
                vert_one_line_[i][grid].insert(vert_one_line_[i][grid].end(),
                                               line_v.begin(),
                                               line_v.end());
            }
            for (const auto& line : edge_line[i])
            {
                size_t              grid   = line.first;
                const vector<Edge>& line_e = line.second;
                edge_one_line_[i][grid].insert(edge_one_line_[i][grid].end(),
                                               line_e.begin(),
                                               line_e.end());
            }
        }
    }

    return 0;
}

int Loop::set_edge_parallel(const Edge& e)
{
    int direction[2];
    for (size_t itr = 0; itr < 2; ++itr)
    {
        size_t axis_d  = (e.get_axis() + itr + 1) % 3;
        direction[itr] = e.get_direction(axis_d);
    }

    size_t id_edge_grid_1 = numeric_limits<size_t>::max();
    size_t id_edge_grid_2 = numeric_limits<size_t>::max();
    if (e.get_edge_on_grid(id_edge_grid_1, id_edge_grid_2))
    {
        if (id_edge_grid_1 != numeric_limits<size_t>::max())
        {
            assert(id_edge_grid_2 == numeric_limits<size_t>::max());
            size_t itr = 0;
            edge_parallel_[itr][id_edge_grid_1].push_back(e);
        }
        else
        {
            assert(id_edge_grid_2 != numeric_limits<size_t>::max());
            size_t itr = 1;
            edge_parallel_[itr][id_edge_grid_2].push_back(e);
        }
    }

    return 0;
}

int Loop::set_invalid_edge()
{
    for (size_t itr = 0; itr < 2; ++itr)
    {
        const auto& edge_parallel_axis = edge_parallel_[itr];
        for (const auto& one_line : edge_parallel_axis)
        {
            get_one_line_invalid_edge(one_line, itr);
        }
    }

    for (size_t itr = 0; itr < 2; ++itr)
    {
        const auto& edge_parallel_axis = edge_invalid_one_line_[itr];
        for (const auto& one_line : edge_parallel_axis)
        {
            size_t grid_cutting = one_line.first;
            for (const auto& e : one_line.second)
            {
                const vector<size_t>& v_line = e.get_line_vert_id();
                vert_invalid_one_line_[itr][grid_cutting].insert(
                    v_line.begin(), v_line.end());
            }
        }
    }

    vector<MatrixXd> table_e;
    for (size_t itr = 0; itr < 2; ++itr)
    {
        for (auto one : edge_invalid_one_line_[itr])
        {
            for (auto e : one.second)
            {
                table_e.push_back(e.get_line());
            }
        }
    }

    return 0;
}

int Loop::get_one_line_invalid_edge(
    const std::pair<size_t, std::vector<Edge>>& one_line,
    size_t                                      itr)
{
    size_t grid_cutting = one_line.first;

    for (const auto& e : one_line.second)
    {
        size_t grid_1 = numeric_limits<size_t>::max();
        size_t grid_2 = numeric_limits<size_t>::max();
        e.get_edge_on_grid(grid_1, grid_2);
        if (itr == 0)
            assert(grid_1 == grid_cutting);
        else
            assert(grid_2 == grid_cutting);

        size_t axis_cutting        = (e.get_axis() + itr + 1) % 3;
        bool   is_next_edge        = true;
        Edge   next_e              = ring_->get_next_edge(e);
        int    direction_cutting_n = next_e.get_direction(axis_cutting);
        while (direction_cutting_n == 0)
        {
            next_e              = ring_->get_next_edge(next_e);
            direction_cutting_n = next_e.get_direction(axis_cutting);
        }
        get_next_invalid_edge(e, next_e, itr, grid_cutting, is_next_edge);

        is_next_edge             = false;
        Edge front_e             = ring_->get_front_edge(e);
        int  direction_cutting_f = front_e.get_direction(axis_cutting);
        while (direction_cutting_f == 0)
        {
            front_e             = ring_->get_front_edge(front_e);
            direction_cutting_f = front_e.get_direction(axis_cutting);
        }
        get_next_invalid_edge(e, front_e, itr, grid_cutting, is_next_edge);
    }

    return 0;
}

int Loop::get_next_invalid_edge(
    const Edge& e,
    const Edge& next_e,
    size_t      itr,
    size_t      grid_cutting,
    bool        is_next_edge)
{
    double p[3] = { 0, 0, 0 };

    size_t axis_intersection = e.get_axis();
    size_t grid_intersection = e.get_grid_line();
    p[axis_intersection]     = e.mesh_->get_grid_line(
        axis_intersection, grid_intersection);

    size_t axis_cutting                  = (e.get_axis() + itr + 1) % 3;
    p[axis_cutting]                      = e.mesh_->get_grid_line(axis_cutting, grid_cutting);
    size_t          axis_projection      = (axis_intersection + 2 - itr) % 3;
    int             direction_projection = e.get_direction(axis_projection);
    const MatrixXd& aabb_m               = e.mesh_->get_aabb();
    if (direction_projection < 0)
    {
        if (is_next_edge)
            p[axis_projection] = aabb_m(axis_projection, 1);
        else
            p[axis_projection] = aabb_m(axis_projection, 0);
    }
    else if (direction_projection > 0)
    {
        if (is_next_edge)
            p[axis_projection] = aabb_m(axis_projection, 0);
        else
            p[axis_projection] = aabb_m(axis_projection, 1);
    }
    else
    {
        assert(false);
    }

    size_t           id_tri   = next_e.get_tri_id();
    vector<Vector3d> v_tri    = e.mesh_->get_tri(id_tri);
    int              is_above = is_vert_above_triangle(
        &v_tri[0](0), &v_tri[1](0), &v_tri[2](0), p);
    if (is_above == -1)  // if vertex a below triangle, then vertex b is valid
        edge_invalid_one_line_[itr][grid_cutting].push_back(next_e);

    assert(is_above != 0);
    return 0;
}

int Loop::set_grid_edge()
{
    set_grid_vert_one_line();

    vector<vector<size_t>> polygon_origin = ring_->get_vertex_sequence();

    size_t itr = 0;
    assert(vert_sorted_[itr].size() == edge_sorted_[itr].size());
    assert(vert_grid_one_line_[itr].size() == vert_sorted_[itr].size());
    size_t num_line = vert_sorted_[itr].size();
    for (size_t i = 0; i < num_line; ++i)
    {
        set_cutted_polygon(itr, i, polygon_origin);
    }
    polygon_cutted_.push(polygon_origin);
    itr = 1;
    assert(vert_sorted_[itr].size() == edge_sorted_[itr].size());
    assert(vert_grid_one_line_[itr].size() == vert_sorted_[itr].size());

    while (!polygon_cutted_.empty())
    {
        vector<vector<size_t>> polygon_origin_cutted = polygon_cutted_.top();
        vector<MatrixXd>       table_p;
        for (auto& p : polygon_origin_cutted)
        {
            MatrixXd poly(3, p.size());
            for (size_t i = 0; i < p.size(); ++i)
            {
                poly.col(i) = ring_->mesh_->get_vert_ptr(p.at(i))->get_vert_coord();
            }
            table_p.push_back(poly);
        }

        polygon_cutted_.pop();
        size_t num_grid = vert_grid_one_line_[itr].size();
        for (size_t i = 0; i < num_grid; ++i)
        {
            set_inner_polygon(itr, i, polygon_origin_cutted);
        }
        polygon_.insert(polygon_.end(),
                        polygon_origin_cutted.begin(),
                        polygon_origin_cutted.end());
    }

    set_inner_polygon_lattice_id();
    return 0;
}

int Loop::set_inner_polygon_lattice_id()
{
    for (const auto& poly : polygon_)
    {
        Vector3st lattice_lower = ring_->get_sequence_lattice(poly);
        Vector3st lattice_upper = lattice_lower;
        lattice_upper(ring_->axis_) += 1;
        vector<size_t> poly_reverse;
        reverse_copy(poly.begin(), poly.end(), back_inserter(poly_reverse));

        polygon_id_.push_back(lattice_lower);

        ring_->mesh_->add_polygon(lattice_lower, poly);
        ring_->mesh_->add_polygon(lattice_upper, poly_reverse);
    }

    return 0;
}

int Loop::set_inner_polygon(
    size_t                  itr,
    size_t                  i,
    vector<vector<size_t>>& polygon_origin)
{
    auto ptr_line = vert_grid_one_line_[itr].begin();
    advance(ptr_line, i);
    const vector<size_t>& vert_one_line_all = ptr_line->second;

    vector<size_t> id_line_v = get_polygon_boundary_line_vert(
        vert_one_line_all, polygon_origin);

    assert(id_line_v.size() % 2 == 0);
    size_t num_v = id_line_v.size();
    for (size_t t = 0; t < num_v; t += 2)  // one valid, next invalid
    {
        size_t         id_prev_v    = id_line_v[t];
        size_t         id_end_v     = id_line_v[t + 1];
        vector<size_t> v_section_id = { id_prev_v, id_end_v };
        get_inner_polygon(itr, id_line_v, v_section_id, t, polygon_origin);
    }

    return 0;
}

int Loop::set_cutted_polygon(
    size_t                  itr,
    size_t                  i,
    vector<vector<size_t>>& polygon_origin)
{
    auto ptr_vert = vert_sorted_[itr].begin();
    auto ptr_edge = edge_sorted_[itr].begin();
    advance(ptr_vert, i);
    advance(ptr_edge, i);

    const vector<size_t>& id_line_v = ptr_vert->second;
    const auto&           edge_line = ptr_edge->second;
    size_t                id_grid   = ptr_vert->first;
    assert(id_line_v.size() % 2 == 0);
    size_t num_v = id_line_v.size();
    for (size_t t = 0; t < num_v; t += 2)  // one valid, next invalid
    {
        size_t      id_prev_v = id_line_v[t];
        size_t      id_end_v  = id_line_v[t + 1];
        const Edge& e_prev    = edge_line[t];
        const Edge& e_end     = edge_line[t + 1];
        if (is_same_vert(id_prev_v, e_prev, id_end_v, e_end))
            continue;

        vector<size_t> v_section_id =
            get_section_vert(itr, id_grid, id_prev_v, id_end_v);

        get_inner_polygon(itr, id_line_v, v_section_id, t, polygon_origin);
    }

    return 0;
}

int Loop::get_section_grid_edge(
    size_t                itr,
    size_t                id_grid,
    size_t                t,
    const vector<size_t>& id_tri,
    const vector<size_t>& id_line_v,
    const vector<Edge>&   e_line,
    vector<Edge>&         edge_grid,
    vector<size_t>&       v_section_id)
{
    size_t   axis            = ring_->axis_;
    size_t   axis_cut        = (axis + itr + 1) % 3;
    size_t   axis_projection = (axis + 2 - itr) % 3;
    Vector3d new_v;
    new_v(axis)     = ring_->mesh_->get_grid_line(axis, ring_->grid_);
    new_v(axis_cut) = ring_->mesh_->get_grid_line(axis_cut, id_grid);

    size_t id_lower = numeric_limits<size_t>::max();
    size_t id_upper = numeric_limits<size_t>::max();
    get_grid_aabb(id_tri[t], id_tri[t + 1], axis_projection, id_lower, id_upper);
    assert(id_lower != numeric_limits<size_t>::max());
    assert(id_upper != numeric_limits<size_t>::max());

    size_t id_prev_v = id_line_v[t];
    size_t id_end_v  = id_line_v[t + 1];

    v_section_id.clear();
    v_section_id.push_back(id_prev_v);
    while (id_lower <= id_upper)
    {
        double p               = ring_->mesh_->get_grid_line(axis_projection, id_lower);
        new_v(axis_projection) = p;
        const Edge& e1         = e_line[t];
        const Edge& e2         = e_line[t + 1];
        // change <= 0 to < 0, filter the vert on triangle
        if (e1.is_vert_above_edge(new_v) < 0 && e2.is_vert_above_edge(new_v) < 0)
        {
            // TODO: error, order is not correct;
            Vert* ptr_new_v = new VertGrid(
                axis, ring_->grid_, axis_cut, id_grid, axis_projection, id_lower);
            size_t id_new_v = ring_->mesh_->add_vert(ptr_new_v);
            ptr_new_v->set_id(id_new_v);
            Edge e(id_prev_v, id_new_v);
            edge_grid.push_back(e);
            id_prev_v = id_new_v;

            v_section_id.push_back(id_new_v);
        }
        ++id_lower;
    }

    v_section_id.push_back(id_end_v);
    Edge e(id_prev_v, id_end_v);
    edge_grid.push_back(e);
    return 0;
}

int Loop::get_grid_aabb(
    size_t  id_tri_1,
    size_t  id_tri_2,
    size_t  axis_projection,
    size_t& id_lower,
    size_t& id_upper)
{
    vector<Vector3d> tri_v_1 = ring_->mesh_->get_tri(id_tri_1);
    vector<Vector3d> tri_v_2 = ring_->mesh_->get_tri(id_tri_2);

    double aabb_tri_axis[2] = { numeric_limits<double>::max(),
                                -numeric_limits<double>::max() };
    for (size_t q = 0; q < 3; ++q)
    {
        aabb_tri_axis[0] = min(aabb_tri_axis[0], tri_v_1[q](axis_projection));
        aabb_tri_axis[0] = min(aabb_tri_axis[0], tri_v_2[q](axis_projection));
        aabb_tri_axis[1] = max(aabb_tri_axis[1], tri_v_1[q](axis_projection));
        aabb_tri_axis[1] = max(aabb_tri_axis[1], tri_v_2[q](axis_projection));
    }

    const MatrixXd& grid_line = ring_->mesh_->get_grid_line();
    const size_t    num_col   = grid_line.cols();
    vector<double>  axis_grid_line(num_col);
    for (size_t j = 0; j < num_col; ++j)
        axis_grid_line[j] = grid_line(axis_projection, j);

    auto ptr_lower = lower_bound(
        axis_grid_line.begin(), axis_grid_line.end(), aabb_tri_axis[0]);
    auto ptr_upper = lower_bound(
        axis_grid_line.begin(), axis_grid_line.end(), aabb_tri_axis[1]);

    id_lower = distance(axis_grid_line.begin(), ptr_lower);
    id_upper = id_lower;
    if (ptr_upper == axis_grid_line.end())
        id_upper = distance(axis_grid_line.begin(), ptr_upper) - 1;
    else
        id_upper = distance(axis_grid_line.begin(), ptr_upper);

    return 0;
}

const array<map<size_t, std::vector<Edge>>, 2>& Loop::get_grid_edge() const
{
    return grid_edge_;
}

int Loop::remove_duplicate_vert_on_one_line()
{
    for (size_t itr = 0; itr < 2; ++itr)
    {
        map<size_t, vector<size_t>>& vert_axis = vert_one_line_[itr];
        map<size_t, vector<Edge>>&   edge_axis = edge_one_line_[itr];
        assert(vert_axis.size() == edge_axis.size());
        const size_t num_grid_line = vert_axis.size();
        for (size_t i = 0; i < num_grid_line; ++i)
        {
            auto ptr_vert = vert_axis.begin();
            auto ptr_edge = edge_axis.begin();
            advance(ptr_vert, i);
            advance(ptr_edge, i);
            vector<size_t>& vert_line = ptr_vert->second;
            vector<Edge>&   edge_line = ptr_edge->second;
            remove_duplicate_vert(itr, vert_line, edge_line);
        }
    }

    return 0;
}

int Loop::remove_duplicate_vert(
    size_t          itr,
    vector<size_t>& vert_line,
    vector<Edge>&   edge_line)
{
    auto ptr_v = vert_line.begin();
    auto ptr_e = edge_line.begin();
    while (ptr_v != vert_line.end())
    {
        auto ptr_next_v = ptr_v + 1;
        auto ptr_next_e = ptr_e + 1;
        if (ptr_next_v == vert_line.end())
            ptr_next_v = vert_line.begin();
        if (ptr_next_e == edge_line.end())
            ptr_next_e = edge_line.begin();

        bool is_same_v = is_same_vert(*ptr_v, *ptr_e, *ptr_next_v, *ptr_next_e);
        if (is_same_v)
        {
            size_t axis = (ring_->axis_ + itr + 1) % 3;
            int    d_e1 = ptr_e->get_direction(axis);
            int    d_e2 = ptr_next_e->get_direction(axis);
            if (d_e1 * d_e2 > 0)
            {
                ptr_v = vert_line.erase(ptr_v);
                ptr_e = edge_line.erase(ptr_e);
                continue;
            }
        }

        ++ptr_v;
        ++ptr_e;
    }
    return 0;
}

bool Loop::is_same_vert(size_t id_v1, const Edge& e1, size_t id_v2, const Edge& e2)
{
    const Vert* ptr_v1 = ring_->mesh_->get_vert_ptr(id_v1);
    const Vert* ptr_v2 = ring_->mesh_->get_vert_ptr(id_v2);
    if (ptr_v1->is_on_triangle_edge() == false
        || ptr_v2->is_on_triangle_edge() == false)
        return false;
    // TODO: on same edge check not finished, finished

    if (ptr_v1->is_on_triangle_vert() && ptr_v2->is_on_triangle_vert())
    {
        if (e1.is_vert_above_edge(id_v2) == 0
            && e2.is_vert_above_edge(id_v1) == 0)
            return true;
        else
            return false;
    }
    else if (ptr_v1->is_on_triangle_vert() || ptr_v2->is_on_triangle_vert())
    {
        return false;
    }
    else
    {
        assert(ptr_v1->is_on_triangle_vert() == false
               && ptr_v2->is_on_triangle_vert() == false);
        assert(ptr_v1->is_on_triangle_edge() && ptr_v2->is_on_triangle_edge());
        size_t v1_e[2] = { numeric_limits<size_t>::max(),
                           numeric_limits<size_t>::max() };
        size_t v2_e[2] = { numeric_limits<size_t>::max(),
                           numeric_limits<size_t>::max() };
        ptr_v1->get_edge_v(v1_e[0], v2_e[0]);
        ptr_v2->get_edge_v(v1_e[1], v2_e[1]);
        for (size_t i = 0; i < 2; ++i)
        {
            assert(v1_e[i] != numeric_limits<size_t>::max());
            assert(v2_e[i] != numeric_limits<size_t>::max());
        }
        if (v1_e[0] == v1_e[1] && v2_e[0] == v2_e[1])
            return true;
        else if (v1_e[0] == v2_e[1] && v2_e[0] == v1_e[1])
            return true;
        else
            return false;
    }
}

int Loop::sort_line_vert()
{
    for (size_t itr = 0; itr < 2; ++itr)
    {
        const size_t num_grid = vert_one_line_[itr].size();
        assert(vert_one_line_[itr].size() == edge_one_line_[itr].size());
        for (size_t i = 0; i < num_grid; ++i)
        {
            sort_one_line(itr, i);
        }
    }

    for (size_t itr = 0; itr < 2; ++itr)
    {
        const size_t num_grid = vert_one_line_[itr].size();
        assert((vert_one_line_[itr].size() == edge_one_line_[itr].size())
               && (vert_one_line_[itr].size() == vert_sorted_[itr].size())
               && (vert_one_line_[itr].size() == edge_sorted_[itr].size()));
        for (size_t i = 0; i < num_grid; ++i)
        {
            auto ptr_v_one_line = vert_one_line_[itr].begin();
            auto ptr_e_one_line = edge_one_line_[itr].begin();
            auto ptr_v_sorted   = vert_sorted_[itr].begin();
            auto ptr_e_sorted   = edge_sorted_[itr].begin();
            advance(ptr_v_one_line, i);
            advance(ptr_e_one_line, i);
            advance(ptr_v_sorted, i);
            advance(ptr_e_sorted, i);
            assert(ptr_v_one_line->second.size() == ptr_v_sorted->second.size());
            assert(ptr_e_one_line->second.size() == ptr_e_sorted->second.size());
            assert(ptr_v_one_line->second.size() == ptr_e_one_line->second.size());
        }
    }

    return 0;
}

int Loop::sort_one_line(size_t itr, size_t i)
{
    auto ptr_v_one_line = vert_one_line_[itr].begin();
    auto ptr_e_one_line = edge_one_line_[itr].begin();
    advance(ptr_v_one_line, i);
    advance(ptr_e_one_line, i);

    const vector<size_t>& vert_line = ptr_v_one_line->second;
    const vector<Edge>&   edge_line = ptr_e_one_line->second;
    assert(vert_line.size() == edge_line.size());
    assert(vert_line.size() % 2 == 0);

    vector<size_t> ordered_seq(vert_line.size());
    iota(ordered_seq.begin(), ordered_seq.end(), 0);

    size_t id_grid = ptr_v_one_line->first;
    sort(ordered_seq.begin(), ordered_seq.end(), [this, &vert_line, &edge_line, itr, id_grid](size_t order_1, size_t order_2) {
        if (order_1 == order_2)
            return false;

        size_t      id_v1 = vert_line[order_1];
        size_t      id_v2 = vert_line[order_2];
        const Edge& e1    = edge_line[order_1];
        const Edge& e2    = edge_line[order_2];
        return is_vert_front(itr, id_grid, id_v1, e1, id_v2, e2);
    });

    update_sorted_vert(itr, id_grid, ordered_seq, vert_line, edge_line);
    return 0;
}

int Loop::update_sorted_vert(
    size_t                itr,
    size_t                id_grid,
    const vector<size_t>& ordered_seq,
    const vector<size_t>& vert_line,
    const vector<Edge>&   edge_line)
{
    vector<size_t> vert_sorted_one_line;
    vector<Edge>   edge_sorted_one_line;
    transform(ordered_seq.begin(), ordered_seq.end(), back_inserter(vert_sorted_one_line), [&vert_line](auto itr) {
        return vert_line[itr];
    });
    transform(ordered_seq.begin(), ordered_seq.end(), back_inserter(edge_sorted_one_line), [&edge_line](auto itr) {
        return edge_line[itr];
    });

    vert_sorted_[itr][id_grid] = vert_sorted_one_line;
    edge_sorted_[itr][id_grid] = edge_sorted_one_line;
    return 0;
}

bool Loop::is_vert_front(size_t itr, size_t id_grid, size_t id_v1, const Edge& e1, size_t id_v2, const Edge& e2)
{
    double p[3];
    int    is_front_aabb = is_vert_front_from_aabb(itr, id_v1, e1, id_v2, e2, p);
    if (is_front_aabb != -1)
        return is_front_aabb;

    constexpr size_t max_sub = 100;
    for (size_t i = 0; i < max_sub; ++i)
    {
        vector<int> inside_v1 = get_vert_inside(itr, id_grid, p, e1);
        vector<int> inside_v2 = get_vert_inside(itr, id_grid, p, e2);
        vector<int> inside_change_v1;
        vector<int> inside_change_v2;
        vector<int> idx_change;
        for (size_t j = 1; j < 3; ++j)
        {
            inside_change_v1.push_back(inside_v1[j] - inside_v1[j - 1]);
            inside_change_v2.push_back(inside_v2[j] - inside_v2[j - 1]);
        }
        for (size_t k = 0; k < 2; ++k)
        {
            if (inside_change_v1[k] != 0 && inside_change_v2[k] == 0)
                return true;
            if (inside_change_v1[k] == 0 && inside_change_v2[k] != 0)
                return false;
            if (inside_change_v1[k] != 0 && inside_change_v2[k] != 0)
                idx_change.push_back(k);
        }
        assert(idx_change.size() == 1);
        update_p(idx_change.front(), p);
    }

    assert(false);
    return false;
}

int Loop::update_p(size_t pos, double* p)
{
    if (pos == 0)
    {
        p[2] = p[1];
        p[1] = (p[0] + p[1]) / 2.0;
        assert(p[0] < p[1] && p[1] < p[2]);
    }
    else
    {
        assert(pos == 1);
        p[0] = p[1];
        p[1] = (p[1] + p[2]) / 2.0;
        assert(p[0] < p[1] && p[1] < p[2]);
    }

    return 0;
}
int Loop::is_vert_front_from_aabb(size_t      itr,
                                  size_t      id_v1,
                                  const Edge& e1,
                                  size_t      id_v2,
                                  const Edge& e2,
                                  double*     p)
{
    bool is_same_v = is_same_vert(id_v1, e1, id_v2, e2);
    if (is_same_v)
    {
        return 0;
    }

    size_t axis_projection = (ring_->axis_ + 2 - itr) % 3;
    size_t id_tri[2]       = { e1.get_tri_id(), e2.get_tri_id() };
    double aabb_tri_axis[2][2];
    for (size_t i = 0; i < 2; ++i)
    {
        vector<Vector3d> tri_v = ring_->mesh_->get_tri(id_tri[i]);
        aabb_tri_axis[i][0]    = numeric_limits<double>::max();
        aabb_tri_axis[i][1]    = -numeric_limits<double>::max();
        for (size_t v = 0; v < 3; ++v)
        {
            aabb_tri_axis[i][0] = min(aabb_tri_axis[i][0],
                                      tri_v[v](axis_projection));
            aabb_tri_axis[i][1] = max(aabb_tri_axis[i][1],
                                      tri_v[v](axis_projection));
        }
        assert(aabb_tri_axis[i][0] != numeric_limits<double>::max());
        assert(aabb_tri_axis[i][1] != -numeric_limits<double>::max());
        assert(aabb_tri_axis[i][0] <= aabb_tri_axis[i][1]);
    }

    if (aabb_tri_axis[0][1] <= aabb_tri_axis[1][0])
        return 1;
    else if (aabb_tri_axis[0][0] >= aabb_tri_axis[1][1])
        return 0;

    p[0] = min(aabb_tri_axis[0][0], aabb_tri_axis[1][0]);
    p[2] = max(aabb_tri_axis[0][1], aabb_tri_axis[1][1]);
    p[1] = (p[0] + p[2]) / 2.0;

    assert(p[0] < p[1] && p[1] < p[2]);
    return -1;
}

vector<int> Loop::get_vert_inside(
    size_t      itr,
    size_t      id_grid,
    double*     p,
    const Edge& e)
{
    size_t   axis            = ring_->axis_;
    size_t   axis_cut        = (axis + itr + 1) % 3;
    size_t   axis_projection = (axis + 2 - itr) % 3;
    Vector3d v;
    v(axis)     = ring_->mesh_->get_grid_line(axis, ring_->grid_);
    v(axis_cut) = ring_->mesh_->get_grid_line(axis_cut, id_grid);

    vector<int> inside_v;
    for (size_t i = 0; i < 3; ++i)
    {
        v(axis_projection) = p[i];
        inside_v.push_back(e.is_vert_above_edge(v));
    }

    return inside_v;
}

const std::array<std::map<size_t, std::vector<Edge>>, 2>&
Loop::get_parallel_grid_edge() const
{
    return edge_parallel_;
}
