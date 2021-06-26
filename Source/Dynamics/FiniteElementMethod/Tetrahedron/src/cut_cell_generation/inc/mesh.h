#ifndef MESH_JJ_H
#define MESH_JJ_H

#include <vector>
#include <Eigen/Core>
#include <map>
#include <unordered_map>
#include "comp.inc"

#include "vert.h"
#include "ring.h"
#include "edge.h"

class Vert;
class Ring;
class Edge;
class Mesh;

struct IdEqualLattice
{
  bool operator()(const Vector3st &id1, const Vector3st &id2) const
    {
      return (id1 == id2);
    }
};

struct HashFucLatticeId
{
  size_t operator()(const Vector3st &id1) const
    {
      return ((std::hash<size_t>()(id1(0))
               ^ (std::hash<size_t>()(id1(1)) << 1)) >> 1)
               ^ (std::hash<size_t>()(id1(2)) << 1);
    }
};

enum PatchType
{
  wall = 1,
  io = 3
};

class Triangle
{
public:
  Triangle();
  int sort_vert();
  int add_cutted_edge(const Edge &e);
  int add_parallel_grid_edge(size_t axis, const Edge &e);
  int set_id(size_t id);
  int cut();
  int write_patch_to_file(const char *const path = "patch") const;

public:
  static int init_triangle(Mesh *mesh);
  int set_triangle_coplanar(size_t axis, size_t p);
  bool is_triangle_coplanar();
  
private:
  int set_vert_on_edge(size_t *id_v);
  int sort_edge_vert();
  bool v_comp(size_t itr, size_t v1, size_t v2);
  int set_cutted_vert();
  int remove_grid_edge();
  int set_direction();
  bool is_cross_vert_front(size_t itr, size_t *axis, size_t *grid);
  int is_vert_on_left(size_t itr, size_t *axis, size_t *grid);
  std::vector<size_t> get_lower_polygon(
    size_t id_prev, size_t id_end, std::vector<size_t> &p);
  int get_sequence_start_end_vert(
    size_t &id_prev, size_t &id_end,
    const std::vector<size_t> &line, const std::vector<size_t> &p);
  size_t get_front_axis(int is_left, int d1, int d2, size_t axis_projection);
  bool is_parallel_vert_front(int d, size_t grid1, size_t grid2);
  std::vector<size_t> get_line_section(
    size_t id_prev, size_t id_end, const std::vector<size_t> &line);


private:
  std::array<std::map<size_t, std::vector<size_t>>, 3> vert_cutted_;
  std::array<std::vector<size_t>, 3> vert_on_edge_sorted_;
  std::array<std::vector<size_t>, 3> vert_on_edge_;

  std::array<std::map<size_t, Edge>, 3> edge_cutted_;
  std::array<std::map<size_t, Edge>, 3> edge_parallel_;
  Eigen::Vector3i direction_[3];

  std::array<size_t, 3> v_id_tri_;
  size_t id_;

  std::vector<std::vector<size_t>> patch_;
  std::vector<Vector3st> patch_id_;
  bool is_coplanar_;
  size_t axis_coplanar_;
  size_t axis_p_;
  
private:
  static Mesh *mesh_;
};


class Mesh
{
public:
  Mesh();
  Mesh(const char *const path);
  ~Mesh();

  friend Triangle;
public:
  int read_from_file(const char *const path);
  size_t get_tri_num() const;
  size_t get_vert_num() const;
  int set_cut_num(size_t num_span);
  Eigen::MatrixXd get_aabb(const std::vector<size_t> &table_id) const;
  const Eigen::MatrixXd &get_aabb() const;

  bool is_same_vert(size_t v1, size_t v2);
  
  int add_polygon(const Vector3st &id_lattice,
                  const std::vector<size_t> &polygon);
  int add_polygon_front(const Vector3st &id_lattice,
                        const std::vector<size_t> &polygon);


  int add_patch(
    const Vector3st &id_lattice, const std::pair<PatchType, size_t> &patch);

  int write_cell_to_file(const char* const path = "cell.vtk");
  int set_cut_line();
  const Vert &get_vert(size_t id_v) const;
  const Vert *get_vert_ptr(size_t id_v) const;
  Vert *get_vert_ptr_to_add_info(size_t id_v);
  double get_grid_line(size_t axis, size_t id_grid) const;
  const std::vector<Eigen::Vector3d> get_tri(size_t id_tri) const;
  const std::vector<Eigen::Vector2d> get_projection_tri(
    size_t id_tri, size_t axis) const;

  const Vector3st &get_tri_vert_id(size_t id_tri) const;
  const Eigen::MatrixXd &get_grid_line() const;
  Eigen::Vector3d get_grid_vert_coordinate(size_t a1, size_t a2, size_t a3) const;
  
  int cut_mesh();

  size_t add_vert(Vert * const vert);
  // size_t add_vert(size_t id_grid1, size_t id_grid2, size_t id_grid3);
  
  int is_vert_inside_triangle(
    size_t id_tri, size_t axis,
    size_t axis_1, size_t grid_1, size_t axis_2, size_t grid_2,
    size_t &v1_e, size_t &v2_e);

  int write_triangle_to_file(
    size_t id_tri, const char* const path = "tri.vtk") const;

  int add_triangle_cutted_edge(size_t id_tri, const Edge &e);
  int add_triangle_parallel_grid_edge(size_t id_tri, size_t itr, const Edge &e);
  int cut_surface();
  int set_tri_coplanar(size_t id, size_t axis, size_t p);
  bool is_triangle_normal_positive(size_t axis_projection, size_t id_tri);
  

  size_t get_axis_grid(
    size_t axis_itr, const std::vector<size_t> &sequence,
    const std::vector<double> &grid_axis);

private:
  size_t get_line_vert_axis_grid_id(
    size_t axis_itr, const Vert *ptr_v,
    const std::vector<double> &grid_axis, std::vector<size_t> &grid_id);

  size_t get_line_vert_grid_id(
    size_t axis_itr, double *p, size_t id_tri,
    const std::vector<double> &grid_axis, std::vector<size_t> &grid_id);

  size_t get_edge_vert_axis_grid_id(
    size_t axis_itr, size_t id_v,
    const std::vector<double> &grid_axis, std::vector<size_t> &grid_id);

  size_t set_projection_edge_vert_axis_grid_id(
    size_t axis_projection, size_t id_v, size_t axis,
    const std::vector<double> &grid_axis, size_t id_grid_line,
    std::vector<size_t> &grid_id, const Eigen::Vector2d v_projection[2]);

private:
  std::vector<Ring> cut_mesh_on_axis_at_p(size_t axis, size_t p);
  int cut_edge(size_t id_tri, Edge &edge);
  std::vector<size_t> cut_triangle_on_axis_at_p(
    size_t id_tri, size_t axis, size_t p);

  int cut_one_tri(
    double lower_t, double upper_t, size_t axis, size_t p, double p_grid,
    size_t id_tri, Ring &ring_lower, Ring &ring_upper, bool &is_coplanar);

private:
  std::vector<Vert*> table_vert_ptr_;
  std::vector<Vector3st> table_tri_;
  std::vector<PatchType> table_tri_type_;
  std::vector<size_t> table_tri_group_id_;

  std::vector<Triangle> table_triangle_;

  Eigen::MatrixXd aabb_;
  size_t num_span_;
  Eigen::MatrixXd grid_line_;

  std::unordered_map<Vector3st, std::list<std::vector<size_t>>,
                     HashFucLatticeId, IdEqualLattice> polygon_cell_;
  std::unordered_map<Vector3st, std::vector<std::pair<PatchType, size_t>>,
                     HashFucLatticeId, IdEqualLattice> patch_cell_;
};


#endif // MESH_JJ_H
