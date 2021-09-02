#ifndef LOOP_JJ_H
#define LOOP_JJ_H

#include <list>
#include <vector>
#include <map>
#include <array>
#include <memory>
#include <set>
#include <stack>

#include "edge.h"
#include "vert.h"

class Vert;
class Edge;
class Mesh;
class Loop;

class Ring
{
public:
    Ring(size_t axis, size_t p);
    Ring() {}

    friend class Loop;

public:
    int    add_edge(const Edge& edge);
    int    add_edge(const std::vector<Edge>& table_e);
    int    sort_to_closed_ring();
    size_t get_edge_num() const;
    int    write_to_file(const char* const path = "ring.vtk") const;
    int    write_edge_to_file(const char* const path = "edge.vtk") const;
    int    remove_duplicate_edge();
    int    add_tri_id(size_t id_tri);
    int    add_coplanar_tri_id(size_t id_tri);
    bool   is_coplanar_tri(size_t id_tri) const;

    int         set_loop(bool need_set_triangle);
    const Edge& get_next_edge(const Edge& e) const;
    const Edge& get_front_edge(const Edge& e) const;
    Vector3st   get_sequence_lattice(const std::vector<size_t>& sequence);

public:
    static int init_ring(Mesh* mesh);

private:
    std::vector<std::vector<size_t>> get_vertex_sequence() const;
    int                              set_vertex_sequence();

    size_t get_axis_grid(
        size_t                     itr_axis,
        const std::vector<size_t>& sequence,
        const std::vector<double>& grid_axis);

    size_t get_line_vert_axis_grid_id(
        size_t                     itr_axis,
        const Vert*                ptr_v,
        const std::vector<double>& grid_axis,
        std::vector<size_t>&       grid_id);

    size_t get_line_vert_grid_id(
        size_t                     itr_axis,
        double*                    p,
        size_t                     id_tri,
        const std::vector<double>& grid_axis,
        std::vector<size_t>&       grid_id);

    size_t get_edge_vert_axis_grid_id(
        size_t                     itr_axis,
        size_t                     id_v,
        const std::vector<double>& grid_axis,
        std::vector<size_t>&       grid_id);

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

class Loop
{
public:
    Loop(Ring* ring);

public:
    int set_grid_line_id();
    int set_vert_on_one_line();
    int set_grid_edge();

    const std::array<std::map<size_t, std::vector<Edge>>, 2>& get_grid_edge() const;
    int                                                       remove_duplicate_vert_on_one_line();
    int                                                       sort_line_vert();
    int                                                       write_polygon_to_file(const char* const path = "inner_polygon.vtk") const;

    const std::array<std::map<size_t, std::vector<Edge>>, 2>&
    get_parallel_grid_edge() const;

private:
    int remove_duplicate_vert(
        size_t               itr,
        std::vector<size_t>& vert_line,
        std::vector<Edge>&   edge_line);
    bool is_same_vert(size_t id_v1, const Edge& e1, size_t id_v2, const Edge& e2);
    int  set_edge_parallel(const Edge& e);

private:
    int get_one_line_invalid_edge(
        const std::pair<size_t, std::vector<Edge>>& one_line,
        size_t                                      itr);
    int set_invalid_edge();
    int get_next_invalid_edge(
        const Edge& e,
        const Edge& next_e,
        size_t      itr,
        size_t      grid_cutting,
        bool        is_next_edge);

    size_t next_vert_of_v_on_polygon(
        size_t                     id_v,
        const std::vector<size_t>& v_on_line,
        const std::vector<size_t>& poly_vert);

    bool is_vertex_on_same_boundary(
        size_t                                  id_prev_v,
        size_t                                  id_end_v,
        const std::vector<std::vector<size_t>>& polygon_origin);

    int connect_boundary(
        const std::vector<size_t>&        v_section_id,
        size_t                            id_prev_v,
        size_t                            id_end_v,
        std::vector<std::vector<size_t>>& polygon_origin);

    std::vector<size_t> get_vertex_sequence(
        size_t                     v1,
        size_t                     v2,
        const std::vector<size_t>& poly);

    int set_inner_polygon(
        size_t                            itr,
        size_t                            i,
        std::vector<std::vector<size_t>>& polygon_origin);

    int set_cutted_polygon(
        size_t                            itr,
        size_t                            i,
        std::vector<std::vector<size_t>>& polygon_origin);

    int get_inner_polygon(
        size_t                            itr,
        const std::vector<size_t>&        id_line_v,
        const std::vector<size_t>&        v_section_id,
        size_t                            t,
        std::vector<std::vector<size_t>>& polygon_origin);

    int set_grid_vert_on(size_t itr, size_t i);
    int set_grid_vert_one_line();

    std::vector<size_t> get_section_vert(
        size_t itr,
        size_t id_grid,
        size_t id_prev_v,
        size_t id_end_v);

    std::vector<size_t> get_polygon_boundary_line_vert(
        const std::vector<size_t>&              vert_one_line_all,
        const std::vector<std::vector<size_t>>& polygon_origin);
    int                 resort_boundary(const std::vector<size_t>& poly, size_t id_v, std::vector<size_t>& poly_new);
    std::vector<size_t> get_polygon(
        size_t                            id_v,
        std::vector<std::vector<size_t>>& polygon_origin);

private:
    int sort_one_line(size_t itr, size_t i);
    int update_sorted_vert(
        size_t                     itr,
        size_t                     id_grid,
        const std::vector<size_t>& ordered_seq,
        const std::vector<size_t>& vert_line,
        const std::vector<Edge>&   edge_line);
    bool             is_vert_front(size_t itr, size_t id_grid, size_t id_v1, const Edge& e1, size_t id_v2, const Edge& e2);
    int              update_p(size_t pos, double* p);
    int              is_vert_front_from_aabb(size_t      itr,
                                             size_t      id_v1,
                                             const Edge& e1,
                                             size_t      id_v2,
                                             const Edge& e2,
                                             double*     p);
    std::vector<int> get_vert_inside(
        size_t      itr,
        size_t      id_grid,
        double*     p,
        const Edge& e);

    int set_axis_grid_edge_on(size_t itr, size_t i);
    int get_section_grid_edge(
        size_t                     itr,
        size_t                     id_grid,
        size_t                     t,
        const std::vector<size_t>& id_tri,
        const std::vector<size_t>& id_line_v,
        const std::vector<Edge>&   e_line,
        std::vector<Edge>&         edge_grid,
        std::vector<size_t>&       v_section_id);

    int get_grid_aabb(size_t id_tri_1, size_t id_tri_2, size_t axis_projection, size_t& id_lower, size_t& id_upper);
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

#endif  // LOOP_JJ_H
