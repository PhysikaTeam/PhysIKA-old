#ifndef EDGE_JJ_H
#define EDGE_JJ_H

#include <vector>
#include <Eigen/Core>
#include <memory>

#include "vert.h"

class Mesh;

class Edge
{
public:
    Edge(size_t id_tri, size_t id_v1, size_t id_v2, size_t axis, size_t grid_line);
    Edge();
    Edge(size_t id_v1, size_t id_v2);

    friend class Loop;

public:
    size_t                       get_axis() const;
    size_t                       get_grid_line() const;
    int                          add_grid_vert(size_t axis, size_t id_vert);
    int                          get_vert_id(size_t& id_v_1, size_t& id_v_2) const;
    int                          sort_grid_line_vert();
    size_t                       get_tri_id() const;
    size_t                       get_line_vert_num() const;
    std::vector<Eigen::Vector3d> get_line_vert() const;
    std::vector<Eigen::MatrixXd> get_line_edge() const;
    Eigen::MatrixXd              get_line() const;
    const std::vector<size_t>&   get_line_vert_id() const;
    int                          set_edge_on_grid();
    int                          get_direction(size_t axis) const;

    bool get_edge_on_grid(size_t& grid_1, size_t& grid_2) const;

    int is_vert_above_edge(const Vector3st& id_v_grid) const;
    int is_vert_above_edge(const size_t id_v) const;
    int is_vert_above_edge(const Eigen::Vector3d& v) const;

    int    set_id(size_t id);
    size_t get_id() const;

public:
    static int init_edge(Mesh* mesh);

private:
    int set_direction();
    int get_axis_projection(
        size_t  n_d_axis,
        size_t& n_projection_axis,
        int&    n_projection_flag);

    size_t    get_front_axis_from_direction(int is_above);
    bool      is_parallel_vert_front(const Vert* ptr_v1, const Vert* ptr_v2, size_t axis);
    size_t    get_vert_parallel_axis(const Vert* ptr_v1, const Vert* ptr_v2);
    Vector3st get_grid_vert_grid_id(const Vert* ptr_v1, const Vert* ptr_v2);
    bool      is_cross_vert_front(Vert* ptr_v1, Vert* ptr_v2);

    int set_edge_on_grid(size_t itr);

private:
    size_t id_;

protected:
    std::vector<size_t> table_line_vert_[2];
    std::vector<size_t> table_sorted_vert_;
    size_t              id_tri_;
    size_t              axis_;
    size_t              grid_line_;
    size_t              id_edge_v_[2];
    int                 direction_[2];  //
    size_t              on_grid_[2];

    // private:
protected:
    static Mesh* mesh_;
};

#endif  // EDGE_JJ_H
