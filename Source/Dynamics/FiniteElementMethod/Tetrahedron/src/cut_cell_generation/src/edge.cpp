#include "../inc/edge.h"
#include <assert.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <Eigen/Geometry>
#include "../inc/mesh.h"
#include "../inc/is_vert_inside_triangle_2d.h"
#include "../inc/comp.inc"
#include "../inc/write_to_file.h"

using namespace std;
using namespace Eigen;


Mesh *Edge::mesh_ = nullptr;

int Edge::init_edge(Mesh *mesh)
{
  mesh_ = mesh;
  return 0;
}

Edge::Edge(size_t id_v1, size_t id_v2)
{
  direction_[0] = direction_[1] = numeric_limits<int>::max();
  on_grid_[0] = on_grid_[1] = numeric_limits<size_t>::max();

  id_edge_v_[0] = id_v1;
  id_edge_v_[1] = id_v2;

  id_ = numeric_limits<size_t>::max();
  id_tri_ = numeric_limits<size_t>::max();
}

Edge::Edge()
{
  direction_[0] = direction_[1] = numeric_limits<int>::max();
  on_grid_[0] = on_grid_[1] = numeric_limits<size_t>::max();

  id_ = numeric_limits<size_t>::max();
  id_tri_ = numeric_limits<size_t>::max();
}

Edge::Edge(size_t id_tri, size_t id_v1, size_t id_v2, size_t axis, size_t grid_line)
{
  direction_[0] = direction_[1] = numeric_limits<int>::max();
  on_grid_[0] = on_grid_[1] = numeric_limits<size_t>::max();

  id_tri_ = id_tri;
  id_edge_v_[0] = id_v1;
  id_edge_v_[1] = id_v2;
  axis_ = axis;
  grid_line_ = grid_line;

  id_ = numeric_limits<size_t>::max();
}

size_t Edge::get_axis() const
{
  return axis_;
}

size_t Edge::get_grid_line() const
{
  return grid_line_;
}

int Edge::add_grid_vert(size_t axis, size_t id_vert)
{
  size_t itr = (3 + axis - axis_) % 3;
  assert(itr >= 1);
  itr -= 1;
  table_line_vert_[itr].push_back(id_vert);
  return 0;
}

int Edge::get_vert_id(size_t &id_v_1, size_t &id_v_2) const
{
  id_v_1 = id_edge_v_[0];
  id_v_2 = id_edge_v_[1];

  return 0;
}

size_t Edge::get_tri_id() const
{
  return id_tri_;
}

size_t Edge::get_line_vert_num() const
{
  return table_line_vert_[0].size() + table_line_vert_[1].size();
}

vector<Vector3d> Edge::get_line_vert() const
{
  vector<Vector3d> table_v;
  for (auto id_v : table_sorted_vert_)
  {
    const Vert *vert = mesh_->get_vert_ptr(id_v);
    table_v.push_back(vert->get_vert_coord());
  }

  return table_v;
}

vector<MatrixXd> Edge::get_line_edge() const
{
  size_t id_v_1 = id_edge_v_[0];
  size_t id_v_2 = id_edge_v_[1];

  Vector3d v1 = mesh_->get_vert(id_v_1).get_vert_coord();
  Vector3d v2 = mesh_->get_vert(id_v_2).get_vert_coord();

  vector<Vector3d> table_v;
  table_v.push_back(v1);
  for (auto id_v : table_sorted_vert_)
  {
    const Vert *vert = mesh_->get_vert_ptr(id_v);
    table_v.push_back(vert->get_vert_coord());
  }
  table_v.push_back(v2);
  
  vector<MatrixXd> table_e;
  size_t num_v = table_v.size();
  for (size_t itr = 1; itr < num_v; ++itr)
  {
    MatrixXd edge(3, 2);
    edge.col(0) = table_v.at(itr - 1);
    edge.col(1) = table_v.at(itr);
    table_e.push_back(edge);
  }

  return table_e;
}

int Edge::set_direction()
{
  for (size_t itr = 0; itr < 2; ++itr)
  {
    size_t n_d_axis = (axis_ + itr + 1) % 3;
    size_t n_projection_axis = numeric_limits<size_t>::max();
    int n_projection_flag = numeric_limits<int>::max();
    get_axis_projection(n_d_axis, n_projection_axis, n_projection_flag);
    assert(n_projection_axis != numeric_limits<size_t>::max());
    assert(n_projection_flag != numeric_limits<int>::max());
    assert(n_projection_flag != 0);
    
    vector<Vector3d> tri_v = mesh_->get_tri(id_tri_);
    int tri_positive = is_triangle_area_positive(n_projection_axis, tri_v);

    if (tri_positive * n_projection_flag > 0)
      direction_[itr] = 1;
    else if (tri_positive * n_projection_flag < 0)
      direction_[itr] = -1;
    else
      direction_[itr] = 0;
  }

  assert(direction_[0] != numeric_limits<int>::max());
  assert(direction_[1] != numeric_limits<int>::max());
  return 0;
}

int Edge::get_axis_projection(
  size_t n_d_axis, size_t &n_projection_axis, int &n_projection_flag)
{
  // (n_g × n_t) * n_d
  // n_t * (n_d × n_g)

  Vector3i n_g = Vector3i::Constant(0);
  n_g(axis_) = 1;

  Vector3i n_d = Vector3i::Constant(0);
  n_d(n_d_axis) = 1;
  Vector3i n_projection = n_d.cross(n_g);

  int sum = accumulate(&n_projection(0), &n_projection(0) + 3, 0,
                       [](int a, int b)
                       {
                         return abs(a) + abs(b);
                       });
  assert(sum == 1);
  for (size_t itr = 0; itr < 3; ++itr)
  {
    if (n_projection(itr) != 0)
    {
      n_projection_axis = itr;
      n_projection_flag = n_projection(itr);
      break;
    }
  }

  return 0;
}


int Edge::set_edge_on_grid()
{
  for (size_t itr = 0; itr < 2; ++itr)
  {
    if (direction_[itr] == 0)
    {
      set_edge_on_grid(itr);
    }
  }
  
  return 0;
}


int Edge::set_edge_on_grid(size_t itr)
{
  const MatrixXd &grid_line = mesh_->get_grid_line();
  const size_t num_col = grid_line.cols();

  size_t axis_projection = (axis_ + 2 - itr) % 3;
  vector<Vector2d> v_tri_2d = mesh_->get_projection_tri(id_tri_, axis_projection);
  set<Vector2d, Vert2DComp> v_set;
  v_set.insert(v_tri_2d.begin(), v_tri_2d.end());
  vector<Vector2d> v_line(v_set.begin(), v_set.end());
  assert(v_line.size() >= 2);

  double p[3];
  p[axis_] = mesh_->get_grid_line(axis_, grid_line_);
  p[axis_projection] = numeric_limits<double>::max();
  size_t axis_cutting = (axis_ + itr + 1) % 3;
  for (size_t col = 0; col < num_col; ++col)
  {
    p[axis_cutting] = mesh_->get_grid_line(axis_cutting, col);
    double p_2d[2];
    p_2d[0] = p[(axis_projection + 1) % 3];
    p_2d[1] = p[(axis_projection + 2) % 3];
    if (is_triangle_area_positive(
          &v_line[0](0), &v_line[1](0), p_2d) == 0)
    {
      on_grid_[itr] = col;
      break;
    }
  }

  return 0;
}

const vector<size_t> &Edge::get_line_vert_id() const
{
  return table_sorted_vert_;
}

MatrixXd Edge::get_line() const
{
  const Vert *ptr_v1 = mesh_->get_vert_ptr(id_edge_v_[0]);
  const Vert *ptr_v2 = mesh_->get_vert_ptr(id_edge_v_[1]);

  MatrixXd edge(3, 2);
  edge.col(0) = ptr_v1->get_vert_coord();
  edge.col(1) = ptr_v2->get_vert_coord();

  return edge;
}

int Edge::get_direction(size_t axis) const
{
  size_t itr = (3 + axis - axis_) % 3 - 1;
  return direction_[itr];
}

bool Edge::get_edge_on_grid(size_t &grid_1, size_t &grid_2) const
{
  grid_1 = on_grid_[0];
  grid_2 = on_grid_[1];

  if (on_grid_[0] != numeric_limits<size_t>::max()
   || on_grid_[1] != numeric_limits<size_t>::max())
    return true;
  else
    return false;
}

int Edge::set_id(size_t id)
{
  id_ = id;
  return 0;
}

size_t Edge::get_id() const
{
  return id_;
}
