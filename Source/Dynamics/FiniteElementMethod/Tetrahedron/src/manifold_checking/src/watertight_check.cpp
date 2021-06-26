#include "../inc/watertight_check.h"
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <vector>
#include <array>
#include <Eigen/Core>
#include "../inc/hash_key.h"
#include "../inc/half_edge.h"
#include "../inc/SimpleObj.h"
#include "../inc/union_find_set.h"
#include "predicates.h"

using namespace std;
using namespace Eigen;

typedef Eigen::Matrix<size_t, 3, 1> Vector3st;
typedef Eigen::Matrix<size_t, 2, 1> Vector2st;


bool is_mesh_watertight(const char *const path)
{
  string str_path(path);
  if (str_path.find(".") == std::string::npos
      || str_path.substr(str_path.rfind(".")) != ".obj")
  {
    cerr << "only obj format supported" << endl;
    return 0;
  }
  
  ifstream f_in(path);
  if (!f_in)
  {
    cerr << "error in file open" << endl;
    return 0;
  }

  function<size_t(const Vector3d &)> HashFunc3 =
    bind(HashFunc<Vector3d, 3>, std::placeholders::_1);
  function<bool(const Vector3d &, const Vector3d &)> EqualKey3 =
    bind(EqualKey<Vector3d, 3>, std::placeholders::_1, std::placeholders::_2);

  unordered_map<Vector3d, size_t, decltype(HashFunc3), decltype(EqualKey3)>
    map_vert(1, HashFunc3, EqualKey3);
  vector<Vector3d> table_vert;
  map<size_t, size_t> map_old_to_new_vert_id;
  vector<Vector3st> table_tri;
  
  string str_line;
  while (getline(f_in, str_line))
  {
    if (str_line.size() < 2)
      continue;

    if (str_line[0] == 'v'
        && str_line[1] == ' ')
    {
      Vector3d v;
      sscanf(str_line.c_str(),
             "%*s %lf %lf %lf", &v(0), &v(1), &v(2));
      if (!map_vert.count(v))
      {
        map_vert.emplace(v, map_vert.size());
        map_old_to_new_vert_id.emplace(
          map_old_to_new_vert_id.size() + 1, map_vert.size() - 1);
        table_vert.push_back(v);
      }
      else
      {
        map_old_to_new_vert_id.emplace(
          map_old_to_new_vert_id.size() + 1, map_vert.at(v));
      }
      continue;
    }
    else if (str_line[0] == 'f'
             && str_line[1] == ' ')
    {
      Vector3st tri_vert;
      if (str_line.find("/") == string::npos)
        sscanf(str_line.c_str(),
               "%*s %zu %zu %zu",
               &tri_vert(0), &tri_vert(1), &tri_vert(2));
      else
        sscanf(str_line.c_str(),
               "%*s %zu%*s %zu%*s %zu%*s",
               &tri_vert(0), &tri_vert(1), &tri_vert(2));

      for (size_t axis = 0; axis < 3; ++axis)
      {
        assert(map_old_to_new_vert_id.count(tri_vert(axis)));
        tri_vert(axis) = map_old_to_new_vert_id.at(tri_vert(axis)); 
      }
        
      table_tri.push_back(tri_vert);
      continue;
    }
  }

  function<size_t(const Vector2st &)> HashFunc2 =
    bind(HashFunc< Vector2st, 2>, std::placeholders::_1);
  function<bool(const Vector2st &, const Vector2st &)> EqualKey2 =
    bind(EqualKey<Vector2st, 2>, std::placeholders::_1, std::placeholders::_2);
  
  unordered_map<Vector2st, Vector3st, decltype(HashFunc2), decltype(EqualKey2)>
    map_edge(1, HashFunc2, EqualKey2);
  for (const auto &tri : table_tri)
  {
    for (size_t i = 0; i < 3; ++i)
    {
      Vector2st e(tri[i], tri[(i+1) % 3]);
      if (map_edge.count(e))
      {
        return false;
      }
      map_edge.emplace(e, tri);
    }
  }

  for (const auto &tri : table_tri)
  {
    for (size_t i = 0; i < 3; ++i)
    {
      Vector2st e(tri[(i+1) % 3], tri[i]);
      if (!map_edge.count(e))
      {
        return false;
      }
    }
  }

  f_in.close();
  return true;
}

bool vert_manifold(const char *const path)
{
  HalfEdgeMesh he_mesh(path);
  unordered_map<size_t, unordered_set<size_t>> vert_face;
  const size_t num_eddge = he_mesh.get_edge_num();
  for (size_t i = 0; i < num_eddge; ++i)
  {
    array<size_t, 2> endpoint = he_mesh.get_edge_vert_id(i);
    array<size_t, 2> neigh_face = he_mesh.get_edge_neighbor_face(i);
    vert_face[endpoint[0]].insert(neigh_face.begin(), neigh_face.end());
    vert_face[endpoint[1]].insert(neigh_face.begin(), neigh_face.end());
  }

  for (const auto &group : vert_face)
  {
    size_t patch_face_num = group.second.size();
    UnionFindSet union_find_set(patch_face_num);
    size_t i = 0;
    for (auto itr = group.second.begin(); itr != group.second.end(); ++itr, ++i)
    {
      auto jtr = itr;
      ++jtr;
      for (size_t j = i+1; jtr != group.second.end(); ++jtr, ++j)
      {
        auto neigh_face = he_mesh.get_face_neigh_face(*itr);
        if (find(neigh_face.begin(), neigh_face.end(), *jtr) != neigh_face.end())
          union_find_set.set_union(i, j);
      }
    }
    if (union_find_set.get_group().size() != 1)
    {
      cerr << "not manifold mesh, vertex nonmanifold" << endl;
      return false;
    }
  }

  return true;  
}

bool is_normal_outside(const char *const path)
{
  HalfEdgeMesh he_mesh(path);
  MatrixXd aabb = he_mesh.get_aabb();
  array<array<vector<size_t>, 2>, 3> box_vert;
  
  const size_t vert_num = he_mesh.get_vert_num();
  for (size_t i = 0; i < vert_num; ++i)
  {
    const Vector3d &vert = he_mesh.get_vert(i);
    for (size_t axis = 0; axis < 3; ++axis)
      for (size_t k = 0; k < 2; ++k)
      {
        if (vert(axis) == aabb(axis, k))
          box_vert[axis][k].push_back(i);
      }
  }

  for (size_t axis = 0; axis < 3; ++axis)
    for (size_t k = 0; k < 2; ++k)
    {
      assert(!box_vert[axis][k].empty());
      for (auto v : box_vert[axis][k])
      {
        vector<size_t> neigh_face = he_mesh.get_vert_neigh_face(v);
        for (auto f : neigh_face)
        {
          Matrix3d face = he_mesh.get_tri(f);
          int dir = get_triangle_normal_axis_direction(face, axis);
          if ((k - 0.5) * dir < 0)
          {
            return false;
          } 
        }
      }
    }
  
  return true;
}

int get_triangle_normal_axis_direction(const Eigen::Matrix3d &face, size_t axis)
{
  array<array<double, 2>, 3> project_tri;
  for (size_t i = 0; i < 2; ++i)
  {
    size_t p = (axis + i + 1) % 3;
    for (size_t v = 0; v < 3; ++v)
      project_tri[v][i] = face(p, v);
  }

  double dir =orient2d(&project_tri[0][0], &project_tri[1][0], &project_tri[2][0]);
  return dir == 0 ? 0 : dir < 0 ? -1 : 1;    
}

bool is_mesh_volume_negative(const char* const path) {
  SimpleObj obj;
  obj.set_obj(path);
  if (obj.compute_directed_volume("tri", false) < 0) {
    return true;
  } else {
    return false;
  }
}
