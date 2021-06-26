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

int Loop::get_inner_polygon(
  size_t itr, const vector<size_t> &id_line_v, const vector<size_t> &v_section_id,
  size_t t, vector<vector<size_t>> &polygon_origin)
{
  vector<size_t> poly_v;
  for (auto &p : polygon_origin)
    poly_v.insert(poly_v.end(), p.begin(), p.end());

  vector<MatrixXd> table_origin;
  for (auto &p : polygon_origin)
  {
    size_t id_v = 0;
    MatrixXd poly(3, p.size());
    for (auto v : p)
    {
      poly.col(id_v) = ring_->mesh_->get_vert_ptr(v)->get_vert_coord();
      ++id_v;
    }
    table_origin.push_back(poly);
  }
    
  vector<size_t> v_on_line(id_line_v.begin() + t, id_line_v.end());
  for (auto v : v_on_line)
  {
    auto is_find = find_if(poly_v.begin(), poly_v.end(),
                           [v, this](size_t p_v) -> bool
                           {
                             return ring_->mesh_->is_same_vert(v, p_v);
                           });
    assert(is_find != poly_v.end());
  }

  size_t id_prev_v = id_line_v[t];
  size_t id_end_v = id_line_v[t+1];
  if (ring_->mesh_->is_same_vert(id_prev_v, id_end_v))
  {
    return 0;
  } 

  if (!is_vertex_on_same_boundary(id_prev_v, id_end_v, polygon_origin))
  {
    connect_boundary(v_section_id, id_prev_v, id_end_v, polygon_origin);
    cerr << "[ \033[1;35mwarning\033[0m ] " << "connect boundary" << endl;
    return 1;
  }
  vector<size_t> poly_vert = get_polygon(id_prev_v, polygon_origin);
  
  vector<size_t> prev_end_sequence =
    get_vertex_sequence(id_prev_v, id_end_v, poly_vert);
  vector<size_t> end_prev_sequence =
    get_vertex_sequence(id_end_v, id_prev_v, poly_vert);

  copy(v_section_id.begin() + 1, v_section_id.end() - 1,
       back_inserter(end_prev_sequence));
  reverse_copy(v_section_id.begin() + 1, v_section_id.end() - 1,
               back_inserter(prev_end_sequence));
  
  size_t v_next_of_prev_v =
    next_vert_of_v_on_polygon(id_prev_v, v_on_line, poly_vert);
  size_t v_next_of_end_v =
    next_vert_of_v_on_polygon(id_end_v, v_on_line, poly_vert);

  if (itr == 0)
  {
    if (ring_->mesh_->is_same_vert(v_next_of_end_v, id_prev_v))
    {
      polygon_cutted_.emplace(1, end_prev_sequence);
    } 
    else
    {
      polygon_origin.push_back(end_prev_sequence);
    } 

    polygon_origin.push_back(prev_end_sequence);
  }
  else
  {
    if (ring_->mesh_->is_same_vert(v_next_of_prev_v, id_end_v))
    {
      polygon_.push_back(prev_end_sequence);
    }
    else
    {
      polygon_origin.push_back(prev_end_sequence);
    }

    polygon_origin.push_back(end_prev_sequence);
  }
  
  return 0;
}

size_t Loop::next_vert_of_v_on_polygon(
  size_t id_v, const vector<size_t> &v_on_line,
  const vector<size_t> &poly_vert)
{
  auto idx_v = find_if(poly_vert.begin(), poly_vert.end(),
                       [id_v, this] (size_t p_v) -> bool
                       {
                         return ring_->mesh_->is_same_vert(id_v, p_v);
                       });
  assert(idx_v != poly_vert.end());
  int d = distance(poly_vert.begin(), idx_v);
  
  vector<int> distance_line_v;
  for (auto v : v_on_line)
  {
    auto pos = find_if(poly_vert.begin(), poly_vert.end(),
                       [v, this] (size_t p_v) -> bool
                       {
                         return ring_->mesh_->is_same_vert(v, p_v);
                       });
    if (pos == poly_vert.end())
      distance_line_v.push_back(numeric_limits<int>::max());
    else
    {
      int d_v = distance(poly_vert.begin(), pos) - d;
      if (d_v < 0)
        d_v += poly_vert.size();
      if (d_v == 0)
        d_v = numeric_limits<int>::max();

      distance_line_v.push_back(d_v);
    }
  }

  vector<size_t> idx(v_on_line.size());
  iota(idx.begin(), idx.end(), 0);

  auto p_min = min_element(idx.begin(), idx.end(),
                           [&distance_line_v](size_t l, size_t r) -> bool
                           {
                             return distance_line_v[l] < distance_line_v[r];
                           });

  assert(p_min != idx.end());
  return v_on_line.at(distance(idx.begin(), p_min));
}

bool Loop::is_vertex_on_same_boundary(
  size_t id_prev_v, size_t id_end_v, const vector<vector<size_t>> &polygon_origin)
{
  size_t count[2] = {0, 0};

  for (auto &p : polygon_origin)
  {
    auto is_find_prev =
      find_if(p.begin(), p.end(),
              [id_prev_v, this] (size_t p_v) -> bool
              {
                return ring_->mesh_->is_same_vert(id_prev_v, p_v);
              });
    auto is_find_end =
      find_if(p.begin(), p.end(),
              [id_end_v, this] (size_t p_v) -> bool
              {
                return ring_->mesh_->is_same_vert(id_end_v, p_v);
              });
    
    if (is_find_prev != p.end())
      ++count[0];
    if (is_find_end != p.end())
      ++count[1];
    
    if (is_find_prev != p.end()
     && is_find_end  != p.end())
      return true;
  }

  assert(count[0] == 1);
  assert(count[1] == 1);
  return false;
}

int Loop::connect_boundary(
  const vector<size_t> &v_section_id, size_t id_prev_v, size_t id_end_v,
  vector<vector<size_t>> &polygon_origin)
{
  vector<size_t> poly_prev = get_polygon(id_prev_v, polygon_origin);
  vector<size_t> poly_end = get_polygon(id_end_v, polygon_origin);

  vector<size_t> poly_new;

  resort_boundary(poly_prev, id_prev_v, poly_new);
  poly_new.insert(
    poly_new.end(), v_section_id.begin() + 1, v_section_id.end() - 1);

  resort_boundary(poly_end, id_end_v, poly_new);
  reverse_copy(v_section_id.begin() + 1, v_section_id.end() - 1,
               back_inserter(poly_new));
  
  polygon_origin.push_back(poly_new);
  return 0;
}

vector<size_t> Loop::get_vertex_sequence(
  size_t v1, size_t v2, const vector<size_t> &poly)
{
  auto idx_v1 = find_if(poly.begin(), poly.end(),
                        [v1, this] (size_t p_v) -> bool
                        {
                          return ring_->mesh_->is_same_vert(v1, p_v);
                        });
  
  auto idx_v2 = find_if(poly.begin(), poly.end(),
                        [v2, this] (size_t p_v) -> bool
                        {
                          return ring_->mesh_->is_same_vert(v2, p_v);
                        });
  assert(idx_v1 != poly.end());
  assert(idx_v2 != poly.end());

  vector<size_t> sequence;
  if (distance(idx_v1, idx_v2) > 0)
  {
    copy(idx_v1, idx_v2 + 1, back_inserter(sequence));
  } 
  else
  {
    copy(idx_v1, poly.end(), back_inserter(sequence));
    copy(poly.begin(), idx_v2 + 1, back_inserter(sequence));
  }

  return sequence;
}

int Loop::write_polygon_to_file(const char *const path) const
{
  vector<MatrixXd> table_p;
  for (auto &p : polygon_)
  {
    MatrixXd poly(3, p.size());
    size_t itr = 0;
    for (auto v : p)
    {
      Vector3d vert = ring_->mesh_->get_vert(v).get_vert_coord();
      poly.col(itr) = vert;
      ++itr;
    }
    table_p.push_back(poly);
  }

  write_to_vtk(table_p, path);
  return 0;
}

int Loop::set_grid_vert_one_line()
{
  for (size_t itr = 0; itr < 2; ++itr)
  {
    assert(vert_sorted_[itr].size() == edge_sorted_[itr].size());
    size_t num_line = vert_sorted_[itr].size();
    for (size_t i = 0; i < num_line; ++i)
    {
      set_grid_vert_on(itr, i);
    }

  }

  return 0;
}

int Loop::set_grid_vert_on(size_t itr, size_t i)
{
  auto ptr_vert = vert_sorted_[itr].begin();
  auto ptr_edge = edge_sorted_[itr].begin();
  advance(ptr_vert, i);
  advance(ptr_edge, i);

  const vector<size_t> &id_line_v = ptr_vert->second;
  const vector<Edge> &e_line = ptr_edge->second;

  assert(id_line_v.size() % 2 == 0);
  assert(id_line_v.size() == e_line.size());
  assert(ptr_vert->first == ptr_edge->first);
  
  vector<Edge> edge_grid;
  size_t id_grid = ptr_vert->first;

  vector<size_t> id_tri;
  transform(e_line.begin(), e_line.end(),
            back_inserter(id_tri),
            [](const Edge &e)
            {
              return e.get_tri_id();
            });

  // TODO: parallel grid edge, finished
  // if (edge_parallel_[itr].count(id_grid) == 0)

  assert(id_line_v.size() % 2 == 0);
  size_t num_v = id_line_v.size();
  for (size_t t = 0; t < num_v - 1; t += 2) // one valid, next invalid
  {
    vector<size_t> v_section_id;
    get_section_grid_edge(
      itr, id_grid, t, id_tri,id_line_v, e_line, edge_grid, v_section_id);
    copy(v_section_id.begin(), v_section_id.end(),
         back_inserter(vert_grid_one_line_[itr][id_grid]));
  }

  grid_edge_[itr][id_grid] = edge_grid;
  return 0;
}

vector<size_t> Loop::get_section_vert(
  size_t itr, size_t id_grid, size_t id_prev_v, size_t id_end_v)
{
  assert(vert_grid_one_line_[itr].count(id_grid));
  const vector<size_t> &one_line = vert_grid_one_line_[itr].at(id_grid);

  auto idx_prev = find(one_line.begin(), one_line.end(), id_prev_v);
  auto idx_end = find(one_line.begin(), one_line.end(), id_end_v);

  assert(idx_prev != one_line.end());
  assert(idx_end != one_line.end());
  assert(distance(idx_prev, idx_end) >= 0);

  vector<size_t> sequence;
  copy(idx_prev, idx_end + 1, back_inserter(sequence));
  return sequence;
}

vector<size_t> Loop::get_polygon_boundary_line_vert(
  const vector<size_t> &vert_one_line_all,
  const vector<vector<size_t>> &polygon_origin)
{
  vector<size_t> poly_v;
  for (auto &p : polygon_origin)
    poly_v.insert(poly_v.end(), p.begin(), p.end());

  vector<size_t> v_line;
  for (auto v : vert_one_line_all)
  {
    auto is_find = find_if(poly_v.begin(), poly_v.end(),
                           [v, this](size_t p_v) -> bool
                           {
                             return ring_->mesh_->is_same_vert(v, p_v);
                           });
    if (is_find != poly_v.end())
      v_line.push_back(v);
  }

  return v_line;
}

vector<size_t> Loop::get_polygon(
  size_t id_v, vector<vector<size_t>> &polygon_origin)
{
  auto pos_prev =
    find_if(polygon_origin.begin(), polygon_origin.end(),
            [id_v, this](auto &p) -> bool
            {
              auto is_find =
                find_if(p.begin(), p.end(),
                        [id_v, this] (size_t p_v) -> bool
                        {
                          return ring_->mesh_->is_same_vert(id_v, p_v);
                        });
              if (is_find != p.end())
                return true;
              else
                return false;
            });
  assert(pos_prev != polygon_origin.end());
  vector<size_t> poly_prev = *pos_prev;
  polygon_origin.erase(pos_prev);

  return poly_prev;
}

int Loop::resort_boundary(const vector<size_t> &poly, size_t id_v,
                           vector<size_t> &poly_new)
{
  auto idx_v = find_if(poly.begin(), poly.end(),
                       [id_v, this] (size_t p_v) -> bool
                       {
                         return ring_->mesh_->is_same_vert(id_v, p_v);
                       });
  assert(idx_v != poly.end());
  poly_new.insert(poly_new.end(), idx_v, poly.end());
  poly_new.insert(poly_new.end(), poly.begin(), idx_v + 1);
  // + 1 to form complete polygon

  return 0;
}
