#pragma once

#include <vector>
#include <Eigen/Core>
#include <memory>

#include "FEMGeometryTetCutCellGenVert.h"

class Mesh;

/**
 * @brief FEM Geometry Edge
 * 
 */
class Edge
{
public:
    /**
     * @brief Construct a new Edge object
     * 
     * @param id_tri 
     * @param id_v1 
     * @param id_v2 
     * @param axis 
     * @param grid_line 
     */
    Edge(size_t id_tri, size_t id_v1, size_t id_v2, size_t axis, size_t grid_line);

    /**
     * @brief Construct a new Edge object
     * 
     */
    Edge();

    /**
     * @brief Construct a new Edge object
     * 
     * @param id_v1 
     * @param id_v2 
     */
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

    /**
     * @brief Get the edge on grid object
     * 
     * @param grid_1 
     * @param grid_2 
     * @return true 
     * @return false 
     */
    bool get_edge_on_grid(size_t& grid_1, size_t& grid_2) const;

    /**
     * @brief Determine whether the vertex is above the age.
     * 
     * @param id_v_grid D
     * @return int 
     */
    int is_vert_above_edge(const Vector3st& id_v_grid) const;

    /**
     * @brief Determine whether the vertex is above the age.
     * 
     * @param id_v 
     * @return int 
     */
    int is_vert_above_edge(const size_t id_v) const;

    /**
     * @brief Determine whether the vertex is above the age.
     * 
     * @param v 
     * @return int 
     */
    int is_vert_above_edge(const Eigen::Vector3d& v) const;

    /**
     * @brief Set the id object
     * 
     * @param id 
     * @return int 
     */
    int set_id(size_t id);

    /**
     * @brief Get the id object
     * 
     * @return size_t 
     */
    size_t get_id() const;

public:
    /**
     * @brief Initialize the age object.
     * 
     * @param mesh 
     * @return int 
     */
    static int init_edge(Mesh* mesh);

private:
    /**
     * @brief Set the direction object
     * 
     * @return int 
     */
    int set_direction();

    /**
     * @brief Get the axis projection object
     * 
     * @param n_d_axis 
     * @param n_projection_axis 
     * @param n_projection_flag 
     * @return int 
     */
    int get_axis_projection(
        size_t  n_d_axis,
        size_t& n_projection_axis,
        int&    n_projection_flag);

    /**
     * @brief Get the front axis from direction object
     * 
     * @param is_above 
     * @return size_t 
     */
    size_t get_front_axis_from_direction(int is_above);

    /**
     * @brief Determine whether the edge is parallel to the axis 
     * 
     * @param ptr_v1 
     * @param ptr_v2 
     * @param axis 
     * @return true 
     * @return false 
     */
    bool is_parallel_vert_front(const Vert* ptr_v1, const Vert* ptr_v2, size_t axis);

    /**
     * @brief Get the vert parallel axis object
     * 
     * @param ptr_v1 
     * @param ptr_v2 
     * @return size_t 
     */
    size_t get_vert_parallel_axis(const Vert* ptr_v1, const Vert* ptr_v2);

    /**
     * @brief Get the grid vert grid id object
     * 
     * @param ptr_v1 
     * @param ptr_v2 
     * @return Vector3st 
     */
    Vector3st get_grid_vert_grid_id(const Vert* ptr_v1, const Vert* ptr_v2);

    /**
     * @brief Determine whether the edge cross the axis 
     * 
     * @param ptr_v1 
     * @param ptr_v2 
     * @return true 
     * @return false 
     */
    bool is_cross_vert_front(Vert* ptr_v1, Vert* ptr_v2);

    /**
     * @brief Set the edge on grid object
     * 
     * @param itr 
     * @return int 
     */
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