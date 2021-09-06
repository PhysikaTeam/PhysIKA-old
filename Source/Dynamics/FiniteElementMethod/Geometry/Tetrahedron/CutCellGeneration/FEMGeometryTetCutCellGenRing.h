#pragma once

#include <list>
#include <vector>
#include <map>
#include <array>
#include <memory>
#include <set>
#include <stack>

#include "FEMGeometryTetCutCellGenEdge.h"
#include "FEMGeometryTetCutCellGenVert.h"

class Vert;
class Edge;
class Mesh;
class Loop;

/**
 * @brief FEM Geometry Ring
 * 
 */
class Ring
{
public:
    /**
     * @brief Construct a new Ring object
     * 
     * @param axis 
     * @param p 
     */
    Ring(size_t axis, size_t p);
    
    /**
     * @brief Construct a new Ring object
     * 
     */
    Ring() {}

    friend class Loop;

public:
    /**
     * @brief Add a edge
     * 
     * @param edge 
     * @return int 
     */
    int    add_edge(const Edge& edge);

    /**
     * @brief Add a edge
     * 
     * @param table_e 
     * @return int 
     */
    int    add_edge(const std::vector<Edge>& table_e);

    /**
     * @brief Sort the closed ring
     * 
     * @return int 
     */
    int    sort_to_closed_ring();

    /**
     * @brief Get the edge num object
     * 
     * @return size_t 
     */
    size_t get_edge_num() const;

    /**
     * @brief Write the ring data to file
     * 
     * @param path 
     * @return int 
     */
    int    write_to_file(const char* const path = "ring.vtk") const;

    /**
     * @brief Write the edge data to file
     * 
     * @param path 
     * @return int 
     */
    int    write_edge_to_file(const char* const path = "edge.vtk") const;

    /**
     * @brief Remove the duplicate edges
     * 
     * @return int 
     */
    int    remove_duplicate_edge();

    /**
     * @brief Add the triangle id
     * 
     * @param id_tri 
     * @return int 
     */
    int    add_tri_id(size_t id_tri);

    /**
     * @brief Add the coplanar triangle id
     * 
     * @param id_tri 
     * @return int 
     */
    int    add_coplanar_tri_id(size_t id_tri);

    /**
     * @brief Determine whether is coplanar with the triangle
     * 
     * @param id_tri 
     * @return true 
     * @return false 
     */
    bool   is_coplanar_tri(size_t id_tri) const;

    /**
     * @brief Set the loop object
     * 
     * @param need_set_triangle 
     * @return int 
     */
    int         set_loop(bool need_set_triangle);

    /**
     * @brief Get the next edge object
     * 
     * @param e 
     * @return const Edge& 
     */
    const Edge& get_next_edge(const Edge& e) const;

    /**
     * @brief Get the front edge object
     * 
     * @param e 
     * @return const Edge& 
     */
    const Edge& get_front_edge(const Edge& e) const;

    /**
     * @brief Get the sequence lattice object
     * 
     * @param sequence 
     * @return Vector3st 
     */
    Vector3st   get_sequence_lattice(const std::vector<size_t>& sequence);

public:
    /**
     * @brief Initialize the ring object
     * 
     * @param mesh 
     * @return int 
     */
    static int init_ring(Mesh* mesh);

private:
    /**
     * @brief Get the vertex sequence object
     * 
     * @return std::vector<std::vector<size_t>> 
     */
    std::vector<std::vector<size_t>> get_vertex_sequence() const;

    /**
     * @brief Set the vertex sequence object
     * 
     * @return int 
     */
    int                              set_vertex_sequence();

    /**
     * @brief Get the axis grid object
     * 
     * @param itr_axis 
     * @param sequence 
     * @param grid_axis 
     * @return size_t 
     */
    size_t get_axis_grid(
        size_t                     itr_axis,
        const std::vector<size_t>& sequence,
        const std::vector<double>& grid_axis);

    /**
     * @brief Get the line vert axis grid id object
     * 
     * @param itr_axis 
     * @param ptr_v 
     * @param grid_axis 
     * @param grid_id 
     * @return size_t 
     */
    size_t get_line_vert_axis_grid_id(
        size_t                     itr_axis,
        const Vert*                ptr_v,
        const std::vector<double>& grid_axis,
        std::vector<size_t>&       grid_id);

    /**
     * @brief Get the line vert grid id object
     * 
     * @param itr_axis 
     * @param p 
     * @param id_tri 
     * @param grid_axis 
     * @param grid_id 
     * @return size_t 
     */
    size_t get_line_vert_grid_id(
        size_t                     itr_axis,
        double*                    p,
        size_t                     id_tri,
        const std::vector<double>& grid_axis,
        std::vector<size_t>&       grid_id);

    /**
     * @brief Get the edge vert axis grid id object
     * 
     * @param itr_axis 
     * @param id_v 
     * @param grid_axis 
     * @param grid_id 
     * @return size_t 
     */
    size_t get_edge_vert_axis_grid_id(
        size_t                     itr_axis,
        size_t                     id_v,
        const std::vector<double>& grid_axis,
        std::vector<size_t>&       grid_id);

    /**
     * @brief Set the projection edge vert axis grid id object
     * 
     * @param itr_axis 
     * @param id_v 
     * @param axis 
     * @param id_grid_line 
     * @param grid_id 
     * @param v_projection 
     * @return size_t 
     */
    size_t set_projection_edge_vert_axis_grid_id(
        size_t                itr_axis,
        size_t                id_v,
        size_t                axis,
        size_t                id_grid_line,
        std::vector<size_t>&  grid_id,
        const Eigen::Vector2d v_projection[2]);

private:
    std::vector<Edge>                table_edge_;
    std::list<Edge>                  table_edge_no_repeat_;
    std::vector<std::list<Edge>>     table_ring_;
    std::vector<std::vector<size_t>> table_vertex_sequence_;
    std::vector<size_t>              table_coplanar_tri_;

    std::vector<size_t>              table_tri_id_;
    std::vector<std::vector<size_t>> table_ring_tri_id_;
    size_t                           axis_;
    size_t                           grid_;

    std::shared_ptr<Loop> loop_;

    std::map<size_t, Edge> map_next_edge_id_;
    std::map<size_t, Edge> map_front_edge_id_;

private:
    static Mesh* mesh_;
};

/**
 * @brief FEM Geometry Loop
 * 
 */
class Loop
{
public:
    /**
     * @brief Construct a new Loop object
     * 
     * @param ring 
     */
    Loop(Ring* ring);

public:
    /**
     * @brief Set the grid line id object
     * 
     * @return int 
     */
    int set_grid_line_id();
    
    /**
     * @brief Set the vert on one line object
     * 
     * @return int 
     */
    int set_vert_on_one_line();

    /**
     * @brief Set the grid edge object
     * 
     * @return int 
     */
    int set_grid_edge();

    /**
     * @brief Get the grid edge object
     * 
     * @return const std::array<std::map<size_t, std::vector<Edge>>, 2>& 
     */
    const std::array<std::map<size_t, std::vector<Edge>>, 2>& get_grid_edge() const;
    /**
     * @brief Remove the duplicate vertexs on one line
     * 
     * @return int 
     */
    int                                                       remove_duplicate_vert_on_one_line();

    /**
     * @brief Sort the vertexs in the line
     * 
     * @return int 
     */
    int                                                       sort_line_vert();

    /**
     * @brief Write polygon data to file
     * 
     * @param path 
     * @return int 
     */
    int                                                       write_polygon_to_file(const char* const path = "inner_polygon.vtk") const;

    /**
     * @brief Get the parallel grid edge object
     * 
     * @return const std::array<std::map<size_t, std::vector<Edge>>, 2>& 
     */
    const std::array<std::map<size_t, std::vector<Edge>>, 2>&
    get_parallel_grid_edge() const;

private:
    /**
     * @brief Remove the duplicate vertexs
     * 
     * @param itr 
     * @param vert_line 
     * @param edge_line 
     * @return int 
     */
    int remove_duplicate_vert(
        size_t               itr,
        std::vector<size_t>& vert_line,
        std::vector<Edge>&   edge_line);

    /**
     * @brief Determine whether the two vertexs is same.
     * 
     * @param id_v1 
     * @param e1 
     * @param id_v2 
     * @param e2 
     * @return true 
     * @return false 
     */
    bool is_same_vert(size_t id_v1, const Edge& e1, size_t id_v2, const Edge& e2);

    /**
     * @brief Set the edge parallel object
     * 
     * @param e 
     * @return int 
     */
    int  set_edge_parallel(const Edge& e);

private:
    /**
     * @brief Get the one line invalid edge object
     * 
     * @param one_line 
     * @param itr 
     * @return int 
     */
    int get_one_line_invalid_edge(
        const std::pair<size_t, std::vector<Edge>>& one_line,
        size_t                                      itr);

    /**
     * @brief Set the invalid edge object
     * 
     * @return int 
     */
    int set_invalid_edge();

    /**
     * @brief Get the next invalid edge object
     * 
     * @param e 
     * @param next_e 
     * @param itr 
     * @param grid_cutting 
     * @param is_next_edge 
     * @return int 
     */
    int get_next_invalid_edge(
        const Edge& e,
        const Edge& next_e,
        size_t      itr,
        size_t      grid_cutting,
        bool        is_next_edge);

    /**
     * @brief Get the number of vertex on the polygon
     * 
     * @param id_v 
     * @param v_on_line 
     * @param poly_vert 
     * @return size_t 
     */
    size_t next_vert_of_v_on_polygon(
        size_t                     id_v,
        const std::vector<size_t>& v_on_line,
        const std::vector<size_t>& poly_vert);

    /**
     * @brief Determine whether the two vertexs is on the same boundry
     * 
     * @param id_prev_v 
     * @param id_end_v 
     * @param polygon_origin 
     * @return true 
     * @return false 
     */
    bool is_vertex_on_same_boundary(
        size_t                                  id_prev_v,
        size_t                                  id_end_v,
        const std::vector<std::vector<size_t>>& polygon_origin);

    /**
     * @brief Connect the boundry
     * 
     * @param v_section_id 
     * @param id_prev_v 
     * @param id_end_v 
     * @param polygon_origin 
     * @return int 
     */
    int connect_boundary(
        const std::vector<size_t>&        v_section_id,
        size_t                            id_prev_v,
        size_t                            id_end_v,
        std::vector<std::vector<size_t>>& polygon_origin);

    /**
     * @brief Get the vertex sequence object
     * 
     * @param v1 
     * @param v2 
     * @param poly 
     * @return std::vector<size_t> 
     */
    std::vector<size_t> get_vertex_sequence(
        size_t                     v1,
        size_t                     v2,
        const std::vector<size_t>& poly);

    /**
     * @brief Set the inner polygon object
     * 
     * @param itr 
     * @param i 
     * @param polygon_origin 
     * @return int 
     */
    int set_inner_polygon(
        size_t                            itr,
        size_t                            i,
        std::vector<std::vector<size_t>>& polygon_origin);

    /**
     * @brief Set the cutted polygon object
     * 
     * @param itr 
     * @param i 
     * @param polygon_origin 
     * @return int 
     */
    int set_cutted_polygon(
        size_t                            itr,
        size_t                            i,
        std::vector<std::vector<size_t>>& polygon_origin);

    /**
     * @brief Get the inner polygon object
     * 
     * @param itr 
     * @param id_line_v 
     * @param v_section_id 
     * @param t 
     * @param polygon_origin 
     * @return int 
     */
    int get_inner_polygon(
        size_t                            itr,
        const std::vector<size_t>&        id_line_v,
        const std::vector<size_t>&        v_section_id,
        size_t                            t,
        std::vector<std::vector<size_t>>& polygon_origin);

    /**
     * @brief Set the grid vert on object
     * 
     * @param itr 
     * @param i 
     * @return int 
     */
    int set_grid_vert_on(size_t itr, size_t i);

    /**
     * @brief Set the grid vert one line object
     * 
     * @return int 
     */
    int set_grid_vert_one_line();

    /**
     * @brief Get the section vert object
     * 
     * @param itr 
     * @param id_grid 
     * @param id_prev_v 
     * @param id_end_v 
     * @return std::vector<size_t> 
     */
    std::vector<size_t> get_section_vert(
        size_t itr,
        size_t id_grid,
        size_t id_prev_v,
        size_t id_end_v);

    /**
     * @brief Get the polygon boundary line vert object
     * 
     * @param vert_one_line_all 
     * @param polygon_origin 
     * @return std::vector<size_t> 
     */
    std::vector<size_t> get_polygon_boundary_line_vert(
        const std::vector<size_t>&              vert_one_line_all,
        const std::vector<std::vector<size_t>>& polygon_origin);

    /**
     * @brief resort the boundary
     * 
     * @param poly 
     * @param id_v 
     * @param poly_new 
     * @return int 
     */
    int                 resort_boundary(const std::vector<size_t>& poly, size_t id_v, std::vector<size_t>& poly_new);

    /**
     * @brief Get the polygon object
     * 
     * @param id_v 
     * @param polygon_origin 
     * @return std::vector<size_t> 
     */
    std::vector<size_t> get_polygon(
        size_t                            id_v,
        std::vector<std::vector<size_t>>& polygon_origin);

private:
    /**
     * @brief Sort the line
     * 
     * @param itr 
     * @param i 
     * @return int 
     */
    int sort_one_line(size_t itr, size_t i);

    /**
     * @brief Update the sorted vertexs 
     * 
     * @param itr 
     * @param id_grid 
     * @param ordered_seq 
     * @param vert_line 
     * @param edge_line 
     * @return int 
     */
    int update_sorted_vert(
        size_t                     itr,
        size_t                     id_grid,
        const std::vector<size_t>& ordered_seq,
        const std::vector<size_t>& vert_line,
        const std::vector<Edge>&   edge_line);

    /**
     * @brief Determine whether the vertex is front
     * 
     * @param itr 
     * @param id_grid 
     * @param id_v1 
     * @param e1 
     * @param id_v2 
     * @param e2 
     * @return true 
     * @return false 
     */
    bool             is_vert_front(size_t itr, size_t id_grid, size_t id_v1, const Edge& e1, size_t id_v2, const Edge& e2);

    /**
     * @brief Update p value
     * 
     * @param pos 
     * @param p 
     * @return int 
     */
    int              update_p(size_t pos, double* p);

    /**
     * @brief Determine whether the vert is front from aabb
     * 
     * @param itr 
     * @param id_v1 
     * @param e1 
     * @param id_v2 
     * @param e2 
     * @param p 
     * @return int 
     */
    int              is_vert_front_from_aabb(size_t      itr,
                                             size_t      id_v1,
                                             const Edge& e1,
                                             size_t      id_v2,
                                             const Edge& e2,
                                             double*     p);

    /**
     * @brief Get the vert inside object
     * 
     * @param itr 
     * @param id_grid 
     * @param p 
     * @param e 
     * @return std::vector<int> 
     */
    std::vector<int> get_vert_inside(
        size_t      itr,
        size_t      id_grid,
        double*     p,
        const Edge& e);

    /**
     * @brief Set the axis grid edge on object
     * 
     * @param itr 
     * @param i 
     * @return int 
     */
    int set_axis_grid_edge_on(size_t itr, size_t i);

    /**
     * @brief Get the section grid edge object
     * 
     * @param itr 
     * @param id_grid 
     * @param t 
     * @param id_tri 
     * @param id_line_v 
     * @param e_line 
     * @param edge_grid 
     * @param v_section_id 
     * @return int 
     */
    int get_section_grid_edge(
        size_t                     itr,
        size_t                     id_grid,
        size_t                     t,
        const std::vector<size_t>& id_tri,
        const std::vector<size_t>& id_line_v,
        const std::vector<Edge>&   e_line,
        std::vector<Edge>&         edge_grid,
        std::vector<size_t>&       v_section_id);

    /**
     * @brief Get the grid aabb object
     * 
     * @param id_tri_1 
     * @param id_tri_2 
     * @param axis_projection 
     * @param id_lower 
     * @param id_upper 
     * @return int 
     */
    int get_grid_aabb(size_t id_tri_1, size_t id_tri_2, size_t axis_projection, size_t& id_lower, size_t& id_upper);

    /**
     * @brief Set the inner polygon lattice id object
     * 
     * @return int 
     */
    int set_inner_polygon_lattice_id();

private:
    std::array<std::vector<size_t>, 2> table_grid_line_id_;

    std::array<std::map<size_t, std::vector<size_t>>, 2> vert_one_line_;
    std::array<std::map<size_t, std::vector<size_t>>, 2> vert_sorted_;
    std::array<std::map<size_t, std::vector<Edge>>, 2>   edge_sorted_;

    std::array<std::map<size_t, std::vector<Edge>>, 2> edge_one_line_;
    std::array<std::map<size_t, std::vector<Edge>>, 2> grid_edge_;

    std::array<std::map<size_t, std::vector<Edge>>, 2> edge_parallel_;

    Eigen::MatrixXd aabb_;
    Ring*           ring_;

private:
    std::array<std::map<size_t, std::vector<Edge>>, 2> edge_invalid_one_line_;
    std::array<std::map<size_t, std::set<size_t>>, 2>  vert_invalid_one_line_;

    std::stack<std::vector<std::vector<size_t>>> polygon_cutted_;

    std::vector<std::vector<size_t>> polygon_;
    std::vector<Vector3st>           polygon_id_;

    std::vector<std::vector<size_t>>                     polygon_origin_[2];
    std::array<std::map<size_t, std::vector<size_t>>, 2> vert_grid_one_line_;
};
