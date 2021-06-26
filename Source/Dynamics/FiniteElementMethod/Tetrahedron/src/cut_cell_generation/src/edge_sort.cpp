#include "../inc/edge.h"
#include <assert.h>
#include <iostream>
#include "../inc/mesh.h"
#include "../inc/is_vert_inside_triangle_2d.h"
#include "../inc/write_to_file.h"


using namespace std;
using namespace Eigen;

int Edge::sort_grid_line_vert()
{
  set_direction();

  // if (table_line_vert_[0].size() == 9)
  table_sorted_vert_.insert(table_sorted_vert_.end(),
                            table_line_vert_[0].begin(),
                            table_line_vert_[0].end());

  // if (table_line_vert_[1].size() == 9)
  table_sorted_vert_.insert(table_sorted_vert_.end(),
                            table_line_vert_[1].begin(),
                            table_line_vert_[1].end());

  sort(table_sorted_vert_.begin(), table_sorted_vert_.end(),
       [this](size_t id_v1, size_t id_v2)
       {
         if (id_v1 == id_v2)
            return false;

         Vert *ptr_v1 = mesh_->get_vert_ptr_to_add_info(id_v1);
         Vert *ptr_v2 = mesh_->get_vert_ptr_to_add_info(id_v2);
         
         size_t axis_parallel = get_vert_parallel_axis(ptr_v1, ptr_v2);
         if (axis_parallel != numeric_limits<size_t>::max())
           return is_parallel_vert_front(ptr_v1, ptr_v2, axis_parallel);

         return is_cross_vert_front(ptr_v1, ptr_v2);
       });

  return 0;
}

bool Edge::is_cross_vert_front(Vert *ptr_v1, Vert *ptr_v2)
{
  Vector3st id_v_grid = get_grid_vert_grid_id(ptr_v1, ptr_v2);
  int is_above = is_vert_above_edge(id_v_grid);

  const Vector3st &id_v1 = ptr_v1->get_grid_id();
  const Vector3st &id_v2 = ptr_v2->get_grid_id();

  if (is_above == 0)
  {
    size_t axis_front = get_front_axis_from_direction(is_above);
    if (ptr_v1->get_grid_line(axis_front) != numeric_limits<size_t>::max())
    {
      assert(ptr_v2->get_grid_line(axis_front) == numeric_limits<size_t>::max());
      return true;
    }
    else
    {
      assert(ptr_v1->get_grid_line(axis_front) == numeric_limits<size_t>::max());
      return false;
    }
  } 
  else
  {
    size_t axis_front = get_front_axis_from_direction(is_above);
    if (ptr_v1->get_grid_line(axis_front) != numeric_limits<size_t>::max())
    {
      assert(ptr_v2->get_grid_line(axis_front) == numeric_limits<size_t>::max());
      return true;
    }
    else
    {
      assert(ptr_v1->get_grid_line(axis_front) == numeric_limits<size_t>::max());
      return false;
    }
  }
}


int Edge::is_vert_above_edge(const Vector3st &id_v_grid) const
{
  Vector3d v = mesh_->get_grid_vert_coordinate(
    id_v_grid(0), id_v_grid(1), id_v_grid(2));

  vector<Vector3d> tri_v = mesh_->get_tri(id_tri_);
  int is_above = is_vert_above_triangle(
    &tri_v[0](0), &tri_v[1](0), &tri_v[2](0), &v(0));

  return is_above;
}

int Edge::is_vert_above_edge(size_t id_v) const
{
  const Vert *ptr_v = mesh_->get_vert_ptr(id_v);
  assert(ptr_v->is_on_triangle_vert());
  Vector3d v = ptr_v->get_vert_coord();
  
  vector<Vector3d> tri_v = mesh_->get_tri(id_tri_);
  int is_above = is_vert_above_triangle(
    &tri_v[0](0), &tri_v[1](0), &tri_v[2](0), &v(0));

  return is_above;
}

int Edge::is_vert_above_edge(const Vector3d &v) const
{
  vector<Vector3d> tri_v = mesh_->get_tri(id_tri_);
  int is_above = is_vert_above_triangle(
    &tri_v[0](0), &tri_v[1](0), &tri_v[2](0), &v(0));

  return is_above;
  
}

Vector3st Edge::get_grid_vert_grid_id(const Vert *ptr_v1, const Vert *ptr_v2)
{
  const Vector3st &id_v1 = ptr_v1->get_grid_id();
  const Vector3st &id_v2 = ptr_v2->get_grid_id();
  
  Vector3st v;
  v << numeric_limits<size_t>::max(),
    numeric_limits<size_t>::max(),
    numeric_limits<size_t>::max();

  for (size_t axis = 0; axis < 3; ++axis)
  {
    if (id_v1(axis) != numeric_limits<size_t>::max())
      v(axis) = id_v1(axis);
    if (id_v2(axis) != numeric_limits<size_t>::max())
      v(axis) = id_v2(axis);

    assert(v(axis) != numeric_limits<size_t>::max());
  }

  
  return v;
}

size_t Edge::get_vert_parallel_axis(const Vert *ptr_v1, const Vert *ptr_v2)
{
  const Vector3st &id_v1 = ptr_v1->get_grid_id();
  const Vector3st &id_v2 = ptr_v2->get_grid_id();
  for (size_t axis = 0; axis < 3; ++axis)
  {
    if (axis == axis_)
      continue;
    if (id_v1(axis) != numeric_limits<size_t>::max()
        && id_v2(axis) != numeric_limits<size_t>::max())
      return axis;
  }

  return numeric_limits<size_t>::max();
}

bool Edge::is_parallel_vert_front(const Vert *ptr_v1, const Vert *ptr_v2, size_t axis)
{
  const Vector3st &id_v1 = ptr_v1->get_grid_id();
  const Vector3st &id_v2 = ptr_v2->get_grid_id();

  // cerr << "id:" << ptr_v1->get_vert_id() << " " << ptr_v2->get_vert_id() << endl;
  // cerr << id_v1 << endl;
  // cerr << id_v2 << endl;
  
  // cerr << "********************" << endl;
  size_t id_grid_v1 = id_v1(axis);
  size_t id_grid_v2 = id_v2(axis);
  assert(id_grid_v1 != id_grid_v2);
  
  size_t itr = (3 + axis - axis_) % 3 - 1;
  assert(itr < 2);
  int direction = direction_[itr];
  assert(direction != 0);

  if (direction < 0)
    return id_grid_v1 > id_grid_v2;
  if (direction > 0)
    return id_grid_v1 < id_grid_v2;
}

size_t Edge::get_front_axis_from_direction(int is_above)
{
  // assert(is_above != 0);
  size_t itr = numeric_limits<size_t>::max();
  if (is_above < 0)
  {
    if (direction_[0] < 0 && direction_[1] < 0)
      itr = 1;
    if (direction_[0] < 0 && direction_[1] > 0)
      itr = 2;
    if (direction_[0] > 0 && direction_[1] < 0)
      itr = 2;
    if (direction_[0] > 0 && direction_[1] > 0)
      itr = 1;
  }
  else
  {
    if (direction_[0] < 0 && direction_[1] < 0)
      itr = 2;
    if (direction_[0] < 0 && direction_[1] > 0)
      itr = 1;
    if (direction_[0] > 0 && direction_[1] < 0)
      itr = 1;
    if (direction_[0] > 0 && direction_[1] > 0)
      itr = 2;
  }

  assert(itr != numeric_limits<size_t>::max());
  return (axis_ + itr) % 3;
}

