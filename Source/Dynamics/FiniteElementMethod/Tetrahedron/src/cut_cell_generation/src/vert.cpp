#include "../inc/vert.h"
#include <Eigen/Geometry>
#include <iostream>
#include "../inc/mesh.h"

using namespace std;
using namespace Eigen;

Mesh* Vert::mesh_ = nullptr;

Vert::Vert(const Vector3d& v)
{
    v_                   = v;
    id_                  = 0;
    is_on_triangle_edge_ = false;
    is_on_triangle_vert_ = false;
    id_edge_v_[0] = id_edge_v_[1] = numeric_limits<size_t>::max();

    for (size_t axis = 0; axis < 3; ++axis)
        grid_line_(axis) = numeric_limits<size_t>::max();
}

Vert::Vert()
{
    id_                  = 0;
    is_on_triangle_edge_ = false;
    is_on_triangle_vert_ = false;
    id_edge_v_[0] = id_edge_v_[1] = numeric_limits<size_t>::max();

    for (size_t axis = 0; axis < 3; ++axis)
        grid_line_(axis) = numeric_limits<size_t>::max();
}

bool Vert::is_on_triangle_edge() const
{
    return is_on_triangle_edge_;
}

int Vert::set_vert_on_triangle_edge()
{
    is_on_triangle_edge_ = true;
    return 0;
}

bool Vert::is_on_triangle_vert() const
{
    return is_on_triangle_vert_;
}

int Vert::set_vert_on_triangle_vert()
{
    is_on_triangle_vert_ = true;
    return 0;
}

int Vert::init_vert(Mesh* mesh)
{
    mesh_ = mesh;
    return 0;
}

const Vector3d Vert::get_vert_coord() const
{
    return v_;
}

int Vert::set_id(size_t id)
{
    id_ = id;
    return 0;
}

int Vert::set_grid_line(size_t axis, size_t grid)
{
    grid_line_(axis) = grid;
    return 0;
}

bool Vert::is_same_vert_with(size_t id_v) const
{
    return id_v == id_;
}

int Vert::get_edge_vert_info(size_t& id_v1, size_t& id_v2, size_t& axis, size_t& id_grid_line) const
{
    id_v1        = numeric_limits<size_t>::max();
    id_v2        = numeric_limits<size_t>::max();
    axis         = numeric_limits<size_t>::max();
    id_grid_line = numeric_limits<size_t>::max();

    return 0;
}

int VertEdge::get_edge_vert_info(size_t& id_v1, size_t& id_v2, size_t& axis, size_t& id_grid_line) const
{
    id_v1        = id_v_[0];
    id_v2        = id_v_[1];
    axis         = axis_;
    id_grid_line = id_grid_;

    return 0;
}

bool VertEdge::is_same_vert_with(size_t id_v) const
{
    size_t id_v1[2]        = { 0, 0 };
    size_t id_v2[2]        = { 0, 0 };
    size_t axis[2]         = { 0, 0 };
    size_t id_grid_line[2] = { 0, 0 };
    mesh_->get_vert_ptr(id_)->get_edge_vert_info(
        id_v1[0], id_v2[0], axis[0], id_grid_line[0]);
    mesh_->get_vert_ptr(id_v)->get_edge_vert_info(
        id_v1[1], id_v2[1], axis[1], id_grid_line[1]);

    if (axis[0] == axis[1] && id_grid_line[0] == id_grid_line[1] && axis[0] < 3)
    {
        if ((id_v1[0] == id_v1[1] && id_v2[0] == id_v2[1])
            || (id_v1[0] == id_v2[1] && id_v2[0] == id_v1[1]))
        {
            return true;
        }
    }

    return false;
}

size_t Vert::get_grid_line(size_t axis) const
{
    return grid_line_(axis);
}

VertEdge::VertEdge(size_t id_v1, size_t id_v2, size_t axis, size_t id_grid_line)
{
    id_v_[0] = id_v1;
    id_v_[1] = id_v2;
    axis_    = axis;
    id_grid_ = id_grid_line;

    grid_line_(axis_) = id_grid_;
}

VertLine::VertLine(size_t id_tri, size_t axis_1, size_t id_grid_1, size_t axis_2, size_t id_grid_2)
{
    axis_[0]    = axis_1;
    axis_[1]    = axis_2;
    id_grid_[0] = id_grid_1;
    id_grid_[1] = id_grid_2;
    id_tri_     = id_tri;

    grid_line_(axis_[0]) = id_grid_[0];
    grid_line_(axis_[1]) = id_grid_[1];
}

const Vector3d VertEdge::get_vert_coord() const
{
    const Vector3d   v1        = mesh_->get_vert(id_v_[0]).get_vert_coord();
    const Vector3d   v2        = mesh_->get_vert(id_v_[1]).get_vert_coord();
    Vector3d         v_diff    = v2 - v1;
    double           grid_line = mesh_->get_grid_line(axis_, id_grid_);
    constexpr double eps       = 1e-9;
    if (fabs(grid_line - v1(axis_)) < eps)
        return v1;

    double alpha = (grid_line - v1(axis_)) / v_diff(axis_);
    return v1 + alpha * (v2 - v1);
}

const Vector3d VertLine::get_vert_coord() const
{
    vector<Vector3d> table_tri_v = mesh_->get_tri(id_tri_);
    Vector3d         p;
    Vector3d         e1;
    Vector3d         e2;
    for (size_t itr = 0; itr < 2; ++itr)
    {
        p(itr) = mesh_->get_grid_line(axis_[itr], id_grid_[itr])
                 - table_tri_v[0](axis_[itr]);
        e1(itr) = table_tri_v[1](axis_[itr]) - table_tri_v[0](axis_[itr]);
        e2(itr) = table_tri_v[2](axis_[itr]) - table_tri_v[0](axis_[itr]);
    }
    p(2) = e1(2) = e2(2) = 0;

    double           de      = e1.cross(e2).norm();
    double           t_alpha = p.cross(e2).norm();
    double           t_beta  = e1.cross(p).norm();
    constexpr double eps     = 1e-9;
    if (fabs(de) < eps)
        return table_tri_v[0];

    double alpha = t_alpha / de;
    double beta  = t_beta / de;
    return table_tri_v[0]
           + alpha * (table_tri_v[1] - table_tri_v[0])
           + beta * (table_tri_v[2] - table_tri_v[0]);
}

VertGrid::VertGrid(
    size_t axis,
    size_t id_grid_1,
    size_t axis_cut,
    size_t id_grid_2,
    size_t axis_projection,
    size_t id_grid_3)
{
    assert(axis < 3);
    grid_line_(axis)            = id_grid_1;
    grid_line_(axis_cut)        = id_grid_2;
    grid_line_(axis_projection) = id_grid_3;
}

const Vector3st& Vert::get_grid_id() const
{
    return grid_line_;
}

bool VertGrid::is_same_vert_with(size_t id_v) const
{
    const Vert*      vert    = mesh_->get_vert_ptr(id_v);
    const Vector3st& id_grid = vert->get_grid_id();
    for (size_t axis = 0; axis < 3; ++axis)
    {
        if (id_grid(axis) != grid_line_(axis))
            return false;
    }

    return true;
}

const Vector3d VertGrid::get_vert_coord() const
{
    Vector3d p;
    for (size_t axis = 0; axis < 3; ++axis)
        p(axis) = mesh_->get_grid_line(axis, grid_line_(axis));

    return p;
}

bool Vert::is_vert_on_axis_grid(size_t axis, size_t id_grid) const
{
    return grid_line_(axis) == id_grid;
}

vector<size_t> Vert::get_triangle_id() const
{
    return { id_ };
}

vector<size_t> VertEdge::get_triangle_id() const
{
    return { id_v_[0], id_v_[1] };
}

size_t Vert::get_vert_id() const
{
    return id_;
}

int Vert::set_edge_v(size_t v1_e, size_t v2_e)
{
    id_edge_v_[0] = v1_e;
    id_edge_v_[1] = v2_e;

    return 0;
}

int Vert::get_edge_v(size_t& v1_e, size_t& v2_e) const
{
    v1_e = id_edge_v_[0];
    v2_e = id_edge_v_[1];

    return 0;
}

int Vert::get_edge_vert(size_t& v1, size_t& v2) const
{
    v1 = numeric_limits<size_t>::max();
    v2 = numeric_limits<size_t>::max();
    return 0;
}

int VertEdge::get_edge_vert(size_t& v1, size_t& v2) const
{
    v1 = id_v_[0];
    v2 = id_v_[1];
    return 0;
}

int Vert::get_line_vert_info(
    size_t& a1,
    size_t& a2,
    size_t& g1,
    size_t& g2,
    size_t& id_tri) const
{
    a1     = numeric_limits<size_t>::max();
    a2     = numeric_limits<size_t>::max();
    g1     = numeric_limits<size_t>::max();
    g2     = numeric_limits<size_t>::max();
    id_tri = numeric_limits<size_t>::max();

    return 0;
}

int VertLine::get_line_vert_info(
    size_t& a1,
    size_t& a2,
    size_t& g1,
    size_t& g2,
    size_t& id_tri) const
{
    a1     = axis_[0];
    a2     = axis_[1];
    g1     = id_grid_[0];
    g2     = id_grid_[1];
    id_tri = id_tri_;

    return 0;
}

bool VertLine::is_same_vert_with(size_t id_v) const
{
    const Vert* ptr_v = mesh_->get_vert_ptr(id_v);
    size_t      a1;
    size_t      a2;
    size_t      g1;
    size_t      g2;
    size_t      id_tri;
    ptr_v->get_line_vert_info(a1, a2, g1, g2, id_tri);

    if (id_tri == this->id_tri_)
    {
        if (a1 == axis_[0] && a2 == axis_[1]
            && g1 == id_grid_[0] && g2 == id_grid_[1])
            return true;
        if (a1 == axis_[1] && a2 == axis_[0]
            && g1 == id_grid_[1] && g2 == id_grid_[0])
            return true;
    }

    return false;
}

bool Vert::is_triangle_vert() const
{
    return true;
}

bool Vert::is_grid_vert() const
{
    return false;
}

bool VertEdge::is_triangle_vert() const
{
    return false;
}

bool VertLine::is_triangle_vert() const
{
    return false;
}

bool VertGrid::is_triangle_vert() const
{
    return false;
}

bool VertGrid::is_grid_vert() const
{
    return true;
}

bool Vert::is_line_vert() const
{
    return false;
}

bool VertLine::is_line_vert() const
{
    return true;
}

bool Vert::is_edge_vert() const
{
    return false;
}

bool VertEdge::is_edge_vert() const
{
    return true;
}
