#include "../inc/ring.h"
#include <Eigen/Core>
#include <iostream>
#include <cmath>
#include <numeric>
#include "../inc/mesh.h"
#include "../inc/edge.h"
#include "../inc/write_to_file.h"
#include "../inc/is_vert_inside_triangle_2d.h"


using namespace std;
using namespace Eigen;

Mesh *Ring::mesh_ = nullptr;

int Ring::init_ring(Mesh *mesh)
{
  mesh_ = mesh;
  return 0;
}

Ring::Ring(size_t axis, size_t p)
{
  axis_ = axis;
  grid_ = p;
}

int Ring::add_edge(const vector<Edge> &table_e)
{
  table_edge_.insert(table_edge_.end(), table_e.begin(), table_e.end());
  return 0;
}


int Ring::add_edge(const Edge &edge)
{
  table_edge_.push_back(edge);
  return 0;
}

size_t Ring::get_edge_num() const
{
  return table_edge_.size();
}

int Ring::write_edge_to_file(const char *const path) const
{
  vector<Eigen::MatrixXd> table_edge;
  vector<double> table_edge_color;
  int color = 0;
  int num_color = 2;
  for (const auto &e : table_edge_no_repeat_)
    table_edge.push_back(e.get_line());
  
  for (auto &itr : table_edge_color)
    itr = itr / num_color;

  write_to_vtk(table_edge, path, &(table_edge_color[0]));
  return 0;
}

int Ring::write_to_file(const char *const path) const
{
  vector<Eigen::MatrixXd> table_edge;
  vector<double> table_edge_color;
  int color = 0;
  int num_color = 2;
  for (const auto &itr : table_ring_)
  {
    for (const auto &e : itr)
    {
      auto line_edge = e.get_line_edge();
      table_edge.insert(table_edge.end(), line_edge.begin(), line_edge.end());
      // table_edge.push_back(e.get_line());
      // table_edge_color.push_back(color);
      // color = (color + 1) % (num_color + 1);
      const size_t num_line_edge = line_edge.size();
      for (size_t itr_line_edge = 0;
           itr_line_edge < num_line_edge; ++itr_line_edge)
      {
        table_edge_color.push_back(color);
        if (itr_line_edge == num_line_edge - 1)
          continue;
        color = (color + 1) % (num_color + 1);
      }
    }
  }

  if (loop_)
  {
    const auto &grid_e = loop_->get_grid_edge();
    for (size_t itr = 0; itr < 2; ++itr)
    {
      auto table_e = grid_e[itr];
      for (auto ge : table_e)
      {
        for (auto e : ge.second)
        {
          table_edge.push_back(e.get_line());
          table_edge_color.push_back(color);
          color = (color + 1) % (num_color + 1);
        }
      }
    }
  }

  for (auto &itr : table_edge_color)
    itr = itr / num_color;

  write_to_vtk(table_edge, path, &(table_edge_color[0]));
  return 0;
}

// TODO: parallel triangle undealed
int Ring::sort_to_closed_ring()
{
  remove_duplicate_edge();
  size_t id_e = 0;
  for (auto &e : table_edge_no_repeat_)
  {
    e.set_id(id_e);
    ++id_e;
  }

  list<Edge> edge_sorted;
  list<Edge> edge_origin = table_edge_no_repeat_;
  vector<size_t> table_edge_tri;

  if (edge_origin.empty())
    return 0;
  
  size_t id_start = 0;
  size_t id_end = 0;
  edge_origin.back().get_vert_id(id_start, id_end);
  const Vert *vert_start = mesh_->get_vert_ptr(id_start);
  const Vert *vert_end = mesh_->get_vert_ptr(id_end);

  edge_sorted.push_back(edge_origin.back());
  table_edge_tri.push_back(edge_origin.back().get_tri_id());
  edge_origin.pop_back();
  Edge e = edge_sorted.back();

  while (!edge_origin.empty())
  {
    bool is_edge_find = false;
    for (auto it = edge_origin.begin(); it != edge_origin.end(); )
    {
      size_t id_edge_start = 0;
      size_t id_edge_end = 0;
      it->get_vert_id(id_edge_start, id_edge_end);
      if (vert_end->is_same_vert_with(id_edge_start))
      {
        assert(!edge_sorted.empty());
        map_next_edge_id_.emplace(edge_sorted.back().get_id(), *it);
        map_front_edge_id_.emplace(it->get_id(), edge_sorted.back());
        
        edge_sorted.push_back(*it);
        table_edge_tri.push_back(it->get_tri_id());

        id_end = id_edge_end;
        vert_end = mesh_->get_vert_ptr(id_end);
        it = edge_origin.erase(it);
        is_edge_find = true;

        break; // break loop to restart
      }
      if (vert_start->is_same_vert_with(id_edge_end))
      {
        assert(!edge_sorted.empty());
        map_next_edge_id_.emplace(it->get_id(), edge_sorted.front());
        map_front_edge_id_.emplace(edge_sorted.front().get_id(), *it);
        
        edge_sorted.push_front(*it);
        table_edge_tri.push_back(it->get_tri_id());

        id_start = id_edge_start;
        vert_start = mesh_->get_vert_ptr(id_start);
        it = edge_origin.erase(it);
        is_edge_find = true;

        break; // break loop to restart
      }
      ++it;
    }
    if (is_edge_find == false)
    {
      vector<Eigen::MatrixXd> table_edge;
      vector<double> table_edge_color;
      int color = 0;
      int num_color = 2;
      for (const auto &e : edge_sorted)
        table_edge.push_back(e.get_line());
  
      for (auto &itr_color : table_edge_color)
        itr_color = itr_color / num_color;

      assert(!edge_sorted.empty());
      map_next_edge_id_.emplace(edge_sorted.back().get_id(), edge_sorted.front());
      map_front_edge_id_.emplace(edge_sorted.front().get_id(), edge_sorted.back());

      size_t id_v1[2] = {0, 0};
      size_t id_v2[2] = {0, 0};
      edge_sorted.front().get_vert_id(id_v1[0], id_v2[0]);
      edge_sorted.back().get_vert_id(id_v1[1], id_v2[1]);
      const Vert *vert_start = mesh_->get_vert_ptr(id_v1[0]);
      assert(vert_start->is_same_vert_with(id_v2[1]));

      table_ring_.push_back(edge_sorted);
      edge_sorted.clear();

      table_ring_tri_id_.push_back(table_edge_tri);
      table_edge_tri.clear();

      edge_origin.back().get_vert_id(id_start, id_end);
      vert_start = mesh_->get_vert_ptr(id_start);
      vert_end = mesh_->get_vert_ptr(id_end);
  
      edge_sorted.push_back(edge_origin.back());
      table_edge_tri.push_back(edge_origin.back().get_tri_id());
      edge_origin.pop_back();
    }
  }

  assert(!edge_sorted.empty());
  map_next_edge_id_.emplace(edge_sorted.back().get_id(), edge_sorted.front());
  map_front_edge_id_.emplace(edge_sorted.front().get_id(), edge_sorted.back());

  table_ring_.push_back(edge_sorted);
  table_ring_tri_id_.push_back(table_edge_tri);

  assert(map_next_edge_id_.size() == table_edge_no_repeat_.size());
  assert(map_front_edge_id_.size() == table_edge_no_repeat_.size());
  for (const auto &ring : table_ring_)
  {
    for (const auto &e : ring)
      assert(e.get_id() != numeric_limits<size_t>::max());

    size_t id_v1[2] = {0, 0};
    size_t id_v2[2] = {0, 0};
    ring.front().get_vert_id(id_v1[0], id_v2[0]);
    ring.back().get_vert_id(id_v1[1], id_v2[1]);
    const Vert *vert_start = mesh_->get_vert_ptr(id_v1[0]);
    assert(vert_start->is_same_vert_with(id_v2[1]));
  }

  if (table_ring_.size() > 1)
  {
    cerr << "warning:: more than one ring find:" << table_ring_.size() << endl;
  }

  set_vertex_sequence();
  return 0;
}

int Ring::set_vertex_sequence()
{
  for (const auto &ring : table_ring_)
  {
    vector<size_t> one_sequence;
    for (const auto &e : ring)
    {
      size_t id_edge_start = 0;
      size_t id_edge_end = 0;
      e.get_vert_id(id_edge_start, id_edge_end);
      const auto &line_v = e.get_line_vert_id();
      one_sequence.push_back(id_edge_start);
      one_sequence.insert(one_sequence.end(), line_v.begin(), line_v.end());
      one_sequence.push_back(id_edge_end);
    }
    table_vertex_sequence_.push_back(one_sequence);
  }
  
  return 0;
}

vector<vector<size_t>> Ring::get_vertex_sequence() const
{
  return table_vertex_sequence_;
}


int Ring::remove_duplicate_edge()
{
  vector<MatrixXd> table_e;
  table_edge_no_repeat_ = list<Edge>(table_edge_.begin(), table_edge_.end());
  // for (auto itr = table_edge_no_repeat_.begin();
  //      itr != table_edge_no_repeat_.end();)
  // {
  //   bool is_duplicate = false;
  //   auto it = itr;
  //   ++it;
  //   for ( ; it != table_edge_no_repeat_.end(); ++it)
  //   {
  //     size_t id_v1[2];
  //     size_t id_v2[2];
  //     itr->get_vert_id(id_v1[0], id_v2[0]);
  //     it->get_vert_id(id_v1[1], id_v2[1]);
                      
  //     if (id_v1[0] == id_v1[1] && id_v2[0] == id_v2[1])
  //     {
  //       table_e.push_back(itr->get_line());
  //       itr = table_edge_no_repeat_.erase(itr);
  //       is_duplicate = true;
  //       break;
  //     }
  //   }
  //   if (!is_duplicate)
  //     ++itr;
  // }

  for (auto itr = table_edge_no_repeat_.begin();
       itr != table_edge_no_repeat_.end();)
  {
    auto it = itr;
    ++it;
    auto is_duplicate =
      find_if(it, table_edge_no_repeat_.end(),
              [itr] (const Edge &e) -> bool
              {
                size_t v_itr[2];
                size_t v_e[2];
                itr->get_vert_id(v_itr[0], v_itr[1]);
                e.get_vert_id(v_e[0], v_e[1]);

                if (v_itr[0] == v_e[0] && v_itr[1] == v_e[1])
                {
                  return true;
                } 
                else
                  return false;
              });
    if (is_duplicate != table_edge_no_repeat_.end())
    {
      assert(distance(itr, is_duplicate) > 0);
      size_t id_tri_find = is_duplicate->get_tri_id();
      size_t id_tri_itr = itr->get_tri_id();
      assert(id_tri_find != numeric_limits<size_t>::max());
      assert(id_tri_itr != numeric_limits<size_t>::max());
      if (!is_coplanar_tri(id_tri_find))
      {
        table_e.push_back(itr->get_line());
        itr = table_edge_no_repeat_.erase(itr);
        continue;
      }
      else
      {
        table_e.push_back(itr->get_line());
        table_edge_no_repeat_.erase(is_duplicate);
        itr = table_edge_no_repeat_.begin();
        continue;
      }
    }
    
    auto is_anti =
      find_if(it, table_edge_no_repeat_.end(),
              [itr] (const Edge &e) -> bool
              {
                size_t v_itr[2];
                size_t v_e[2];
                itr->get_vert_id(v_itr[0], v_itr[1]);
                e.get_vert_id(v_e[0], v_e[1]);

                if (v_itr[1] == v_e[0] && v_itr[0] == v_e[1])
                  return true;
                else
                  return false;
              });

    if (is_anti != table_edge_no_repeat_.end())
    {
      table_e.push_back(is_anti->get_line());
      Edge e_itr = *itr;
      table_edge_no_repeat_.erase(is_anti);

      auto is_find =
        find_if(table_edge_no_repeat_.begin(), table_edge_no_repeat_.end(),
                [&e_itr] (const Edge &e) -> bool
                {
                  size_t v_itr[2];
                  size_t v_e[2];
                  e_itr.get_vert_id(v_itr[0], v_itr[1]);
                  e.get_vert_id(v_e[0], v_e[1]);

                  if (v_itr[0] == v_e[0] && v_itr[1] == v_e[1])
                    return true;
                  else
                    return false;
                });
      table_e.push_back(is_find->get_line());
      assert(is_find != table_edge_no_repeat_.end());
      itr = table_edge_no_repeat_.erase(is_find);
      continue;
    }

    ++itr;
  }


  if (table_e.size())
    write_to_vtk(table_e, "duplicat_e.vtk");
  return 0;
}

int Ring::add_tri_id(size_t id_tri)
{
  table_tri_id_.push_back(id_tri);
  return 0;
}

int Ring::set_loop(bool need_set_triangle)
{
  assert(table_ring_.size() == table_ring_tri_id_.size());
  loop_ = make_shared<Loop>(this);
  loop_->set_grid_line_id();
  loop_->set_vert_on_one_line();
  // loop_->remove_duplicate_vert_on_one_line();
  loop_->sort_line_vert();
  loop_->set_grid_edge();

  if (need_set_triangle)
  {
    for (auto &e : table_edge_no_repeat_)
      mesh_->add_triangle_cutted_edge(e.get_tri_id(), e);
    const array<map<size_t, vector<Edge>>, 2> &edge_parallel_grid =
      loop_->get_parallel_grid_edge();

    for (size_t itr = 0; itr < 2; ++itr)
      for (const auto &one_line : edge_parallel_grid[itr])
        for (const auto &e : one_line.second)
          mesh_->add_triangle_parallel_grid_edge(e.get_tri_id(), itr, e);

  }

  return 0;
}

const Edge &Ring::get_next_edge(const Edge &e) const
{
  assert(map_next_edge_id_.count(e.get_id()));
  return map_next_edge_id_.at(e.get_id());
}

const Edge &Ring::get_front_edge(const Edge &e) const
{
  assert(map_front_edge_id_.count(e.get_id()));
  return map_front_edge_id_.at(e.get_id());
}

int Ring::add_coplanar_tri_id(size_t id_tri)
{
  table_coplanar_tri_.push_back(id_tri);
  return 0;
}

bool Ring::is_coplanar_tri(size_t id_tri) const
{
  auto is_find =
    find(table_coplanar_tri_.begin(), table_coplanar_tri_.end(), id_tri);

  return (is_find != table_coplanar_tri_.end());
}

Vector3st Ring::get_sequence_lattice(const vector<size_t> &sequence)
{
  Vector3st lattice =
    Matrix<size_t, 3, 1>::Constant(numeric_limits<size_t>::max());

  lattice(axis_) = grid_;

  const MatrixXd &grid_line = mesh_->get_grid_line();
  vector<double> grid_vec[3];
  for (size_t i = 0; i < 3; ++i)
  {
    for (size_t j = 0; j < grid_line.cols(); ++j)
      grid_vec[i].push_back(grid_line(i, j));
  }

  for (size_t itr_axis = 0; itr_axis < 2; ++itr_axis)
  {
    size_t axis = (axis_ + itr_axis + 1) % 3;
    size_t axis_grid = mesh_->get_axis_grid(axis, sequence, grid_vec[axis]);
    lattice(axis) = axis_grid;
  }

  return lattice;
}

