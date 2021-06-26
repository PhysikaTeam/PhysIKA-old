#include "../inc/mesh.h"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <string>
#include <iomanip>
#include "../inc/vert.h"
#include "../inc/ring.h"
#include "../inc/edge.h"
#include "../inc/is_vert_inside_triangle_2d.h"
#include "../inc/write_to_file.h"


using namespace std;
using namespace Eigen;

Mesh *Triangle::mesh_ = nullptr;

Triangle::Triangle()
{
  axis_coplanar_ = numeric_limits<size_t>::max();
  axis_p_ = numeric_limits<size_t>::max();
  is_coplanar_ = false;
}


int Triangle::init_triangle(Mesh *mesh)
{
  mesh_ = mesh;
  return 0;
}

int Triangle::set_id(size_t id)
{
  id_ = id;
  return 0;
}

int Triangle::add_cutted_edge(const Edge &e)
{
  assert(e.get_tri_id() == id_);
  size_t axis = e.get_axis();
  size_t grid_line = e.get_grid_line();

  edge_cutted_[axis][grid_line] = e;
  return 0;
}

int Triangle::add_parallel_grid_edge(size_t axis, const Edge &e)
{
  assert(e.get_tri_id() == id_);
  size_t grid_line = e.get_grid_line();
  edge_parallel_[axis][grid_line] = e;

  return 0;
}

int Triangle::sort_vert()
{
  set_direction();
  remove_grid_edge();
  set_cutted_vert();

  sort_edge_vert();
  return 0;
}

bool Triangle::v_comp(size_t itr, size_t v1, size_t v2)
{
  Vector3i d = direction_[itr];
  const Vert *ptr_v1 = mesh_->get_vert_ptr(v1);
  const Vert *ptr_v2 = mesh_->get_vert_ptr(v2);
  Vector3st grid_line[2] = {ptr_v1->get_grid_id(), ptr_v2->get_grid_id()};

  size_t axis[2] = {numeric_limits<size_t>::max(), numeric_limits<size_t>::max()};
  size_t grid[2] = {numeric_limits<size_t>::max(), numeric_limits<size_t>::max()};
  for (size_t i = 0; i < 2; ++i)
  {
    for (size_t j = 0; j < 3; ++j)
    {
      if (grid_line[i](j) != numeric_limits<size_t>::max())
      {
        axis[i] = j;
        grid[i] = grid_line[i](j);
        break;
      }
    }
    assert(axis[i] != numeric_limits<size_t>::max());
    assert(grid[i] != numeric_limits<size_t>::max());
  }

  if (axis[0] == axis[1])
  {
    size_t a = axis[0];
    return is_parallel_vert_front(d[a], grid[0], grid[1]);
  }
  else
  {
    return is_cross_vert_front(itr, axis, grid);
  }

  assert(false);
}

size_t Triangle::get_front_axis(
  int is_left, int d1, int d2, size_t axis_projection)
{
  const Vector3st &v_tri_id = mesh_->get_tri_vert_id(id_);
  Vector2d v_tri_projected[3];
  for (size_t i = 0; i < 3; ++i)
  {
    Vector3d v = mesh_->get_vert_ptr(v_tri_id(i))->get_vert_coord();
    Vector2d v_p;
    v_p[0] = v((axis_projection + 1) % 3);
    v_p[1] = v((axis_projection + 2) % 3);
    v_tri_projected[i] = v_p;
  }

  int area = is_triangle_area_positive(
    &v_tri_projected[0](0),
    &v_tri_projected[1](0),
    &v_tri_projected[2](0));

  size_t itr = numeric_limits<size_t>::max();
  if (is_left > 0)
  {
    if (d1 < 0 && d2 < 0)
      itr = 1;
    else if (d1 < 0 && d2 > 0)
      itr = 2;
    else if (d1 > 0 && d2 < 0)
      itr = 2;
    else if (d1 > 0 && d2 > 0)
      itr = 1;
  }
  else if (is_left < 0)
  {
    if (d1 < 0 && d2 < 0)
      itr = 2;
    else if (d1 < 0 && d2 > 0)
      itr = 1;
    else if (d1 > 0 && d2 < 0)
      itr = 1;
    else if (d1 > 0 && d2 > 0)
      itr = 2;
  }
  else
  {
    if (area > 0)
    {
    if (d1 < 0 && d2 < 0)
      itr = 2;
    else if (d1 < 0 && d2 > 0)
      itr = 1;
    else if (d1 > 0 && d2 < 0)
      itr = 1;
    else if (d1 > 0 && d2 > 0)
      itr = 2;
    }
    else if (area < 0)
    {
    if (d1 < 0 && d2 < 0)
      itr = 1;
    else if (d1 < 0 && d2 > 0)
      itr = 2;
    else if (d1 > 0 && d2 < 0)
      itr = 2;
    else if (d1 > 0 && d2 > 0)
      itr = 1;
    }
    else
    {
      if (axis_projection == 0)
        return 1;
      else
        return 0;
    }
  }
  
  assert(itr != numeric_limits<size_t>::max());
  return (axis_projection + itr) % 3;
}

int Triangle::sort_edge_vert()
{
  for (size_t itr = 0; itr < 3; ++itr)
  {
    vector<size_t> v_edge = vert_on_edge_[itr];
    // for (auto it = v_edge.begin(); it != v_edge.end();)
    // {
    //   if (find(v_id_tri_.begin(), v_id_tri_.end(), *it) != v_id_tri_.end())
    //   {
    //     it = v_edge.erase(it);
    //     continue;
    //   }
    //   ++it;
    // }
    
    sort(v_edge.begin(), v_edge.end(),
         [itr, this] (size_t v1, size_t v2) -> bool
         {
           return v_comp(itr, v1, v2);
         });
    vert_on_edge_sorted_[itr] = v_edge;
  }

  return 0;
}


int Triangle::set_cutted_vert()
{
  for (size_t axis = 0; axis < 3; ++axis)
  {
    for (const auto &line : edge_cutted_[axis])
    {
      const Edge &e = line.second;
      size_t id_v[2] = {numeric_limits<size_t>::max(),
                        numeric_limits<size_t>::max()};
      e.get_vert_id(id_v[0], id_v[1]);
      assert(id_v[0] != numeric_limits<size_t>::max()
             && id_v[1] != numeric_limits<size_t>::max());
      
      size_t grid_line = line.first;
      const vector<size_t> &v_line = e.get_line_vert_id();

      vert_cutted_[axis][grid_line].push_back(id_v[0]);
      for (auto v : v_line)
      {
        const Vert *ptr_v = mesh_->get_vert_ptr(v);
        if (!ptr_v->is_on_triangle_edge())
          vert_cutted_[axis][grid_line].push_back(v);
      }
      vert_cutted_[axis][grid_line].push_back(id_v[1]);
      
      set_vert_on_edge(id_v);
    }
  }

  return 0;
}

int Triangle::set_vert_on_edge(size_t *id_v)
{
  const Vert *ptr_v1 = mesh_->get_vert_ptr(id_v[0]);
  const Vert *ptr_v2 = mesh_->get_vert_ptr(id_v[1]);
  pair<size_t, size_t> v_tri[2];
      
  ptr_v1->get_edge_vert(v_tri[0].first, v_tri[0].second);
  ptr_v2->get_edge_vert(v_tri[1].first, v_tri[1].second);

  for (size_t i = 0; i < 2; ++i)
  {
    if (v_tri[i].first == numeric_limits<size_t>::max()
        && v_tri[i].second == numeric_limits<size_t>::max())
      continue;
    
    assert(v_tri[i].first != numeric_limits<size_t>::max()
           && v_tri[i].second != numeric_limits<size_t>::max());

    bool is_find = false;
    for (size_t itr = 0; itr < 3; ++itr)
    {
      pair<size_t, size_t> e = {v_id_tri_[itr], v_id_tri_[(itr + 1) % 3]};
      if (e == v_tri[i])
      {
        is_find = true;
        vert_on_edge_[itr].push_back(id_v[i]);
        break;
      }
    }
    assert(is_find);
  }

  return 0;
}

int Triangle::remove_grid_edge()
{
  for (size_t axis = 0; axis < 3; ++axis)
  {
    auto &edge_axis = edge_cutted_[axis];
    for (auto itr = edge_axis.begin(); itr != edge_axis.end(); )
    {
      size_t v_e[2] =
        {numeric_limits<size_t>::max(), numeric_limits<size_t>::max()};
      itr->second.get_vert_id(v_e[0], v_e[1]);
      assert(v_e[0] != numeric_limits<size_t>::max());
      assert(v_e[1] != numeric_limits<size_t>::max());

      bool is_tri_edge = false;
      pair<size_t, size_t> ve = {v_e[0], v_e[1]};
      for (size_t v = 0; v < 3; ++v)
      {
        pair<size_t, size_t> e_tri = {v_id_tri_[v], v_id_tri_[(v + 1) % 3]};
        if (ve == e_tri || (ve.first == e_tri.second && ve.second == e_tri.first))
        {
          itr = edge_axis.erase(itr);
          is_tri_edge = true;
          break;
        }
      }
      if (!is_tri_edge)
        itr++;
    }
  }

  int sum_axis_parallel = 0;
  for (size_t itr = 0; itr < 3; ++itr)
  {
    if (!edge_parallel_[itr].empty())
      ++sum_axis_parallel;
  }
  assert(sum_axis_parallel % 2 <= 2);
  if (sum_axis_parallel == 0)
    return 0;
  
  size_t axis_parallel = 2;
  for (size_t itr = 0; itr < 3; ++itr)
  {
    size_t axis = axis_parallel - itr;
    if (!edge_parallel_[axis].empty())
    {
      axis_parallel = axis;
      break;
    }
  }

  if (is_coplanar_)
    return 0;

  map<size_t, Edge> &edge_parallel_one_axis = edge_parallel_[axis_parallel];
  map<size_t, Edge> &edge_one_axis = edge_cutted_[axis_parallel];
  for (auto itr = edge_one_axis.begin(); itr != edge_one_axis.end(); )
  {
    if (edge_parallel_one_axis.count(itr->first))
    {
      itr = edge_one_axis.erase(itr);
      continue;
    }
    else
      ++itr;
  }


  return 0;
}

int Triangle::set_direction()
{
  const Vector3st &id_tri_v = mesh_->get_tri_vert_id(id_);
  vector<Vector3d> table_v_coord(3);
  for (size_t itr = 0; itr < 3; ++itr)
  {
    v_id_tri_[itr] = id_tri_v(itr);
    const Vert *ptr_v = mesh_->get_vert_ptr(v_id_tri_[itr]);
    table_v_coord[itr] = ptr_v->get_vert_coord();
  } 
  
  for (size_t itr = 0; itr < 3; ++itr)
  {
    Vector3i d;
    for (size_t axis = 0; axis < 3; ++axis)
    {
      if (table_v_coord[itr](axis) < table_v_coord[(itr + 1) % 3](axis))
        d(axis) = 1;
      else if (table_v_coord[itr](axis) > table_v_coord[(itr + 1) % 3](axis))
        d(axis) = -1;
      else
        d(axis) = 0;
    }
    direction_[itr] = d;
  }

  return 0;
}

bool Triangle::is_parallel_vert_front(int d, size_t grid1, size_t grid2)
{
  if (d > 0)
  {
    if (grid1 < grid2)
      return true;
    else
      return false;
  }
  else if (d < 0)
  {
    if (grid1 < grid2)
      return false;
    else
      return true;
  }
  else
  {
    assert(false);
  }
  
}

bool Triangle::is_cross_vert_front(size_t itr, size_t *axis, size_t *grid)
{
  int is_left = is_vert_on_left(itr, axis, grid);
  Vector3i d = direction_[itr];
  size_t axis_projection = 3 - axis[0] - axis[1];
  size_t a1 = (axis_projection + 1) % 3;
  size_t a2 = (axis_projection + 2) % 3;
  size_t axis_front = get_front_axis(is_left, d[a1], d[a2], axis_projection);
  if (axis[0] == axis_front)
  {
    return true;
  }
  else
  {
    assert(axis[1] == axis_front);
    return false;
  }
}

int Triangle::is_vert_on_left(size_t itr, size_t *axis, size_t *grid)
{
  size_t axis_projection = 3 - axis[0] - axis[1];
  double p[2];
  if (axis[0] == (axis_projection + 1) % 3)
  {
    p[0] = mesh_->get_grid_line(axis[0], grid[0]);
    p[1] = mesh_->get_grid_line(axis[1], grid[1]);
  }
  else
  {
    assert(axis[1] == (axis_projection + 1) % 3);
    p[0] = mesh_->get_grid_line(axis[1], grid[1]);
    p[1] = mesh_->get_grid_line(axis[0], grid[0]);
  }

  size_t v_tri[2] = {v_id_tri_[itr], v_id_tri_[(itr + 1) % 3]};
  Vector2d v_projected[2];
  for (size_t i = 0; i < 2; ++i)
  {
    Vector3d coord = mesh_->get_vert_ptr(v_tri[i])->get_vert_coord();
    Vector2d v;
    v(0) = coord((axis_projection + 1) % 3);
    v(1) = coord((axis_projection + 2) % 3);
    v_projected[i] = v;
  }
  
  int is_left = is_triangle_area_positive(
    &v_projected[0](0), &v_projected[1](0), p);

  return is_left;
}

int Triangle::cut()
{
  vector<size_t> vert_sequence;
  for (size_t itr = 0; itr < 3; ++itr)
  {
    vert_sequence.push_back(v_id_tri_[itr]);
    vert_sequence.insert(
      vert_sequence.end(),
      vert_on_edge_sorted_[itr].begin(),
      vert_on_edge_sorted_[itr].end());
  }

  stack<vector<size_t>> polygon[4];
  stack<Vector3st> polygon_id[4];
  polygon[0].push(vert_sequence);

  Vector3st lattice_id_tri =
    Matrix<size_t, 3, 1>::Constant(numeric_limits<size_t>::max());
  Vector3st id_v_tri = mesh_->get_tri_vert_id(id_);
  vector<size_t> v_tri = {id_v_tri(0), id_v_tri(1), id_v_tri(2)};
  MatrixXd aabb_tri = mesh_->get_aabb(v_tri);

  const MatrixXd &grid_line = mesh_->get_grid_line();
  for (size_t axis = 0; axis < 3; ++axis)
  {
    if (is_coplanar_)
    {
      assert(axis_coplanar_ < 3);
      bool is_positive = mesh_->is_triangle_normal_positive(axis_coplanar_, id_);
      if (is_positive)
      {
        lattice_id_tri(axis) = axis_p_; 
      }
      else
      {
        lattice_id_tri(axis) = axis_p_ + 1;
      }
      continue;
    }

    vector<double> axis_grid_line(grid_line.cols());
    for (size_t i = 0; i < grid_line.cols(); ++i)
    {
      axis_grid_line[i] = grid_line(axis, i);
    }
    auto idx = lower_bound(
      axis_grid_line.begin(), axis_grid_line.end(), aabb_tri(axis, 1));
    lattice_id_tri(axis) = distance(axis_grid_line.begin(), idx);
  }
  polygon_id[0].push(lattice_id_tri);

  for (size_t itr = 0; itr < 3; ++itr)
  {
    const auto &axis_vert = vert_cutted_[itr];
    while (!polygon[itr].empty())
    {
      assert(!polygon_id[itr].empty());
      vector<size_t> p = polygon[itr].top();
      Vector3st p_id = polygon_id[itr].top();
      polygon[itr].pop();
      polygon_id[itr].pop();

      for (auto &line : axis_vert)
      {
        size_t id_prev = numeric_limits<size_t>::max();
        size_t id_end = numeric_limits<size_t>::max();
        get_sequence_start_end_vert(id_prev, id_end, line.second, p);

        if (id_prev == numeric_limits<size_t>::max()
          && id_end == numeric_limits<size_t>::max())
          continue;

        assert(!mesh_->is_same_vert(id_prev, id_end));
        vector<size_t> lower_p = get_lower_polygon(id_prev, id_end, p);
        vector<size_t> section_l = get_line_section(id_prev, id_end, line.second);
        reverse_copy(section_l.begin(), section_l.end(), back_inserter(lower_p));
        copy(section_l.begin(), section_l.end(), back_inserter(p));
        
        polygon[itr + 1].push(lower_p);
        p_id(itr) = line.first;
        polygon_id[itr + 1].push(p_id);
        p_id(itr) = line.first + 1;
        
        assert(polygon[itr+1].top().size() != 1);
      }
      polygon[itr + 1].push(p);
      polygon_id[itr + 1].push(p_id);

      assert(polygon[itr+1].top().size() != 1);
    }
    assert(polygon_id[itr].empty());
  }

  vector<double> grid_vec[3];
  for (size_t i = 0; i < 3; ++i)
  {
    for (size_t j = 0; j < grid_line.cols(); ++j)
      grid_vec[i].push_back(grid_line(i, j));
  }
  while (!polygon[3].empty())
  {
    patch_.push_back(polygon[3].top());
    patch_id_.push_back(polygon_id[3].top());

    Vector3st lattice =
      Matrix<size_t, 3, 1>::Constant(numeric_limits<size_t>::max());
    for (size_t axis = 0; axis < 3; ++axis)
    {
      lattice(axis) =
        mesh_->get_axis_grid(axis, polygon[3].top(), grid_vec[axis]);
    }


    const auto poly = polygon[3].top();
    size_t num_v = poly.size();
    for (size_t i = 1; i < num_v - 1; ++i)
    {
      vector<size_t> tri_patch = {poly.at(0), poly.at(i), poly.at(i + 1)};
      mesh_->add_polygon_front(lattice, tri_patch);
      mesh_->add_patch(lattice, make_pair(mesh_->table_tri_type_.at(id_),
                                          mesh_->table_tri_group_id_.at(id_)));
    }

    polygon[3].pop();
    polygon_id[3].pop();
  }
  
  return 0;
}

vector<size_t> Triangle::get_lower_polygon(
  size_t id_prev, size_t id_end, vector<size_t> &p)
{
  // assert(!mesh_->is_same_vert(id_prev, id_end));
  auto find_prev = find_if(p.begin(), p.end(),
                           [id_prev, this] (size_t v) -> bool
                           {
                             return mesh_->is_same_vert(id_prev, v);
                           });
  auto find_end = find_if(p.begin(), p.end(),
                          [id_end, this] (size_t v) -> bool
                          {
                            return mesh_->is_same_vert(id_end, v);
                          });

  vector<size_t> lower_p;
  // if (find_prev == p.end() && find_end == p.end())
  //   return lower_p;

  assert(find_prev != p.end() && find_end != p.end());
  if (distance(find_prev, find_end) > 0)
  {
    lower_p.insert(lower_p.end(), find_prev, find_end + 1);
    
    vector<size_t> new_p;
    new_p.insert(new_p.end(), find_end, p.end());
    new_p.insert(new_p.end(), p.begin(), find_prev + 1);
    p = new_p;


    MatrixXd lower_poly(3, lower_p.size());
    MatrixXd poly(3, p.size());
    for (size_t i = 0; i < lower_p.size(); ++i)
    {
      lower_poly.col(i) = mesh_->get_vert_ptr(lower_p.at(i))->get_vert_coord();
    }
    for (size_t i = 0; i < p.size(); ++i)
    {
      poly.col(i) = mesh_->get_vert_ptr(p.at(i))->get_vert_coord();
    }

    // write_to_vtk(vector<MatrixXd>(1, lower_poly), "lp.vtk");
    // write_to_vtk(vector<MatrixXd>(1, poly), "p.vtk");
    // getchar();

    return lower_p;
  }
  else
  {
    assert(distance(find_prev, find_end) < 0);
    assert(distance(find_end, find_prev) > 0);
    lower_p.insert(lower_p.end(), find_prev, p.end());
    lower_p.insert(lower_p.end(), p.begin(), find_end + 1);

    vector<size_t> new_p;
    new_p.insert(new_p.end(), find_end, find_prev + 1);
    p = new_p;
    MatrixXd lower_poly(3, lower_p.size());
    MatrixXd poly(3, p.size());
    for (size_t i = 0; i < lower_p.size(); ++i)
    {
      lower_poly.col(i) = mesh_->get_vert_ptr(lower_p.at(i))->get_vert_coord();
    }
    for (size_t i = 0; i < p.size(); ++i)
    {
      poly.col(i) = mesh_->get_vert_ptr(p.at(i))->get_vert_coord();
    }
    // write_to_vtk(vector<MatrixXd>(1, lower_poly), "lp.vtk");
    // write_to_vtk(vector<MatrixXd>(1, poly), "p.vtk");
    // getchar();

    return lower_p;
  }
}

int Triangle::write_patch_to_file(const char *const path) const
{
  vector<MatrixXd> table_p;
  for (const auto &p : patch_)
  {
    size_t num_v = p.size();
    MatrixXd poly(3, num_v);
    for (size_t col = 0; col < num_v; ++col)
    {
      const Vert *ptr_v = mesh_->get_vert_ptr(p[col]);
      poly.col(col) = ptr_v->get_vert_coord();
    }
    table_p.push_back(poly);
  }

  write_to_vtk(table_p, path);
  return 0;
}

int Triangle::get_sequence_start_end_vert(
  size_t &id_prev, size_t &id_end,
  const vector<size_t> &line, const vector<size_t> &p)
{
  size_t v_count = 0;
  for (auto v : line)
  {
    auto is_find = find_if(p.begin(), p.end(),
                           [v, this] (size_t p_v) -> bool
                           {
                             return mesh_->is_same_vert(v, p_v);
                           });
    if (is_find != p.end())
      ++v_count;
  }
  if (v_count % 2 == 1)
    return 0;
  
  size_t count = 0;
  for (auto v : line)
  {
    auto is_find = find_if(p.begin(), p.end(),
                           [v, this] (size_t p_v) -> bool
                           {
                             return mesh_->is_same_vert(v, p_v);
                           });
    if (is_find != p.end())
    {
      id_prev = v;
      ++count;
      break;
    }
  }

  for (auto itr = line.rbegin(); itr != line.rend(); ++itr)
  {
    auto is_find = find_if(p.begin(), p.end(),
                           [itr, this](size_t p_v) -> bool
                           {
                             return mesh_->is_same_vert(*itr, p_v);
                           });
    if (is_find != p.end())
    {
      id_end = *itr;
      ++count;
      break;
    }
  }

  assert(count % 2 == 0);
  return 0;
}

vector<size_t> Triangle::get_line_section(
  size_t id_prev, size_t id_end, const vector<size_t> &line)
{
  auto is_find_prev = find_if(line.begin(), line.end(),
                              [id_prev, this] (size_t p_v) -> bool
                              {
                                return mesh_->is_same_vert(id_prev, p_v);
                              });

  auto is_find_end = find_if(line.begin(), line.end(),
                             [id_end, this] (size_t p_v) -> bool
                             {
                               return mesh_->is_same_vert(id_end, p_v);
                             });

  assert(is_find_prev != line.end());
  assert(is_find_end != line.end());
  assert(distance(is_find_prev, is_find_end) > 0);
  if (distance(is_find_prev, is_find_end) < 2)
    return vector<size_t>(0);

  vector<size_t> section_l(is_find_prev + 1, is_find_end);
  return section_l;
}

int Triangle::set_triangle_coplanar(size_t axis, size_t p)
{
  axis_coplanar_ = axis;
  axis_p_ = p;
  
  is_coplanar_ = true;
  return 0;
}

bool Triangle::is_triangle_coplanar()
{
  return is_coplanar_;
}
