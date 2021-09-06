#pragma once

#include <vector>
#include <Eigen/Core>
#include <map>
#include <unordered_map>
#include "FEMGeometryTetCutCellGenComp.h"

#include "FEMGeometryTetCutCellGenVert.h"
#include "FEMGeometryTetCutCellGenRing.h"
#include "FEMGeometryTetCutCellGenVert.h"

class Vert;
class Ring;
class Edge;
class Mesh;

/**
 * @brief FEM Geometry IdEqualLattice.
 * 
 */
struct IdEqualLattice
{
    bool operator()(const Vector3st& id1, const Vector3st& id2) const
    {
        return (id1 == id2);
    }
};

/**
 * @brief FEM Geometry HashFucLatticeId
 * 
 */
struct HashFucLatticeId
{
    size_t operator()(const Vector3st& id1) const
    {
        return ((std::hash<size_t>()(id1(0))
                 ^ (std::hash<size_t>()(id1(1)) << 1))
                >> 1)
               ^ (std::hash<size_t>()(id1(2)) << 1);
    }
};

enum PatchType
{
    wall = 1,
    io   = 3
};

/**
 * @brief FEM Geometry Triangle.
 * 
 */
class Triangle
{
public:
    /**
     * @brief Construct a new Triangle object
     * 
     */
    Triangle();

    /**
     * @brief Sort the vertexs.
     * 
     * @return int 
     */
    int sort_vert();

    /**
     * @brief Add a cutted edge.
     * 
     * @param e 
     * @return int 
     */
    int add_cutted_edge(const Edge& e);

    /**
     * @brief Add a parallel grid edge.
     * 
     * @param axis 
     * @param e 
     * @return int 
     */
    int add_parallel_grid_edge(size_t axis, const Edge& e);

    /**
     * @brief Set the id object
     * 
     * @param id 
     * @return int 
     */
    int set_id(size_t id);

    /**
     * @brief cut 
     * 
     * @return int 
     */
    int cut();

    /**
     * @brief Write patch to the file.
     * 
     * @param path 
     * @return int 
     */
    int write_patch_to_file(const char* const path = "patch") const;

public:
    static int init_triangle(Mesh* mesh);
    int        set_triangle_coplanar(size_t axis, size_t p);
    bool       is_triangle_coplanar();

private:
    int  set_vert_on_edge(size_t* id_v);
    int  sort_edge_vert();
    bool v_comp(size_t itr, size_t v1, size_t v2);
    int  set_cutted_vert();
    int  remove_grid_edge();
    int  set_direction();
    bool is_cross_vert_front(size_t itr, size_t* axis, size_t* grid);
    int  is_vert_on_left(size_t itr, size_t* axis, size_t* grid);
    /**
     * @brief Get the lower polygon object
     * 
     * @param id_prev 
     * @param id_end 
     * @param p 
     * @return std::vector<size_t> 
     */
    std::vector<size_t> get_lower_polygon(
        size_t               id_prev,
        size_t               id_end,
        std::vector<size_t>& p);

    /**
     * @brief Get the sequence start end vert object
     * 
     * @param id_prev 
     * @param id_end 
     * @param line 
     * @param p 
     * @return int 
     */
    int get_sequence_start_end_vert(
        size_t&                    id_prev,
        size_t&                    id_end,
        const std::vector<size_t>& line,
        const std::vector<size_t>& p);

    /**
     * @brief Get the front axis object
     * 
     * @param is_left 
     * @param d1 
     * @param d2 
     * @param axis_projection 
     * @return size_t 
     */
    size_t get_front_axis(int is_left, int d1, int d2, size_t axis_projection);

    /**
     * @brief Determine whether vertexs is parallel.
     * 
     * @param d 
     * @param grid1 
     * @param grid2 
     * @return true 
     * @return false 
     */
    bool is_parallel_vert_front(int d, size_t grid1, size_t grid2);

    /**
     * @brief Get the line section object
     * 
     * @param id_prev 
     * @param id_end 
     * @param line 
     * @return std::vector<size_t> 
     */
    std::vector<size_t> get_line_section(
        size_t                     id_prev,
        size_t                     id_end,
        const std::vector<size_t>& line);

private:
    std::array<std::map<size_t, std::vector<size_t>>, 3> vert_cutted_;
    std::array<std::vector<size_t>, 3>                   vert_on_edge_sorted_;
    std::array<std::vector<size_t>, 3>                   vert_on_edge_;

    std::array<std::map<size_t, Edge>, 3> edge_cutted_;
    std::array<std::map<size_t, Edge>, 3> edge_parallel_;
    Eigen::Vector3i                       direction_[3];

    std::array<size_t, 3> v_id_tri_;
    size_t                id_;

    std::vector<std::vector<size_t>> patch_;
    std::vector<Vector3st>           patch_id_;
    bool                             is_coplanar_;
    size_t                           axis_coplanar_;
    size_t                           axis_p_;

private:
    static Mesh* mesh_;
};

class Mesh
{
public:
    /**
     * @brief Construct a new Mesh object
     * 
     */
    Mesh();

    /**
     * @brief Construct a new Mesh object
     * 
     * @param path 
     */
    Mesh(const char* const path);

    /**
     * @brief Destroy the Mesh object
     * 
     */
    ~Mesh();

    friend Triangle;

public:
    /**
     * @brief Read data from file
     * 
     * @param path 
     * @return int 
     */
    int read_from_file(const char* const path);

    /**
     * @brief Get the tri num object
     * 
     * @return size_t 
     */
    size_t get_tri_num() const;

    /**
     * @brief Get the vert num object
     * 
     * @return size_t 
     */
    size_t get_vert_num() const;

    /**
     * @brief Set the cut num object
     * 
     * @param num_span 
     * @return int 
     */
    int set_cut_num(size_t num_span);

    /**
     * @brief Get the aabb object
     * 
     * @param table_id 
     * @return Eigen::MatrixXd 
     */
    Eigen::MatrixXd get_aabb(const std::vector<size_t>& table_id) const;

    /**
     * @brief Get the aabb object
     * 
     * @return const Eigen::MatrixXd& 
     */
    const Eigen::MatrixXd& get_aabb() const;

    /**
     * @brief Detemine whether the two vertexs is same
     * 
     * @param v1 
     * @param v2 
     * @return true 
     * @return false 
     */
    bool is_same_vert(size_t v1, size_t v2);

    /**
     * @brief Add polygon data
     * 
     * @param id_lattice 
     * @param polygon 
     * @return int 
     */
    int add_polygon(const Vector3st&           id_lattice,
                    const std::vector<size_t>& polygon);

    /**
     * @brief Add polygon front
     * 
     * @param id_lattice 
     * @param polygon 
     * @return int 
     */
    int add_polygon_front(const Vector3st&           id_lattice,
                          const std::vector<size_t>& polygon);

    /**
     * @brief Add patch
     * 
     * @param id_lattice 
     * @param patch 
     * @return int 
     */
    int add_patch(
        const Vector3st&                    id_lattice,
        const std::pair<PatchType, size_t>& patch);

    /**
     * @brief Write cell to file
     * 
     * @param path 
     * @return int 
     */
    int write_cell_to_file(const char* const path = "cell.vtk");

    /**
     * @brief Set the cut line object
     * 
     * @return int 
     */
    int set_cut_line();

    /**
     * @brief Get the vert object
     * 
     * @param id_v 
     * @return const Vert& 
     */
    const Vert& get_vert(size_t id_v) const;

    /**
     * @brief Get the vert ptr object
     * 
     * @param id_v 
     * @return const Vert* 
     */
    const Vert* get_vert_ptr(size_t id_v) const;

    /**
     * @brief Get the vert ptr to add info object
     * 
     * @param id_v 
     * @return Vert* 
     */
    Vert* get_vert_ptr_to_add_info(size_t id_v);

    /**
     * @brief Get the grid line object
     * 
     * @param axis 
     * @param id_grid 
     * @return double 
     */
    double get_grid_line(size_t axis, size_t id_grid) const;

    /**
     * @brief Get the tri object
     * 
     * @param id_tri 
     * @return const std::vector<Eigen::Vector3d> 
     */
    const std::vector<Eigen::Vector3d> get_tri(size_t id_tri) const;

    /**
     * @brief Get the projection tri object
     * 
     * @param id_tri 
     * @param axis 
     * @return const std::vector<Eigen::Vector2d> 
     */
    const std::vector<Eigen::Vector2d> get_projection_tri(
        size_t id_tri,
        size_t axis) const;

    /**
     * @brief Get the tri vert id object
     * 
     * @param id_tri 
     * @return const Vector3st& 
     */
    const Vector3st& get_tri_vert_id(size_t id_tri) const;

    /**
     * @brief Get the grid line object
     * 
     * @return const Eigen::MatrixXd& 
     */
    const Eigen::MatrixXd& get_grid_line() const;

    /**
     * @brief Get the grid vert coordinate object
     * 
     * @param a1 
     * @param a2 
     * @param a3 
     * @return Eigen::Vector3d 
     */
    Eigen::Vector3d get_grid_vert_coordinate(size_t a1, size_t a2, size_t a3) const;

    /**
     * @brief Cut the mesh
     * 
     * @return int 
     */
    int cut_mesh();

    /**
     * @brief add vertexs
     * 
     * @param vert 
     * @return size_t 
     */
    size_t add_vert(Vert* const vert);
    // size_t add_vert(size_t id_grid1, size_t id_grid2, size_t id_grid3);

    /**
     * @brief Determine whether a vertex is inside the triangle
     * 
     * @param id_tri 
     * @param axis 
     * @param axis_1 
     * @param grid_1 
     * @param axis_2 
     * @param grid_2 
     * @param v1_e 
     * @param v2_e 
     * @return int 
     */
    int is_vert_inside_triangle(
        size_t  id_tri,
        size_t  axis,
        size_t  axis_1,
        size_t  grid_1,
        size_t  axis_2,
        size_t  grid_2,
        size_t& v1_e,
        size_t& v2_e);

    /**
     * @brief Write triangle to file
     * 
     * @param id_tri 
     * @param path 
     * @return int 
     */
    int write_triangle_to_file(
        size_t            id_tri,
        const char* const path = "tri.vtk") const;

    /**
     * @brief Add a triangle cutted a edge
     * 
     * @param id_tri 
     * @param e 
     * @return int 
     */
    int add_triangle_cutted_edge(size_t id_tri, const Edge& e);

    /**
     * @brief Add a triangle paralleled to a grid edge 
     * 
     * @param id_tri 
     * @param itr 
     * @param e 
     * @return int 
     */
    int add_triangle_parallel_grid_edge(size_t id_tri, size_t itr, const Edge& e);

    /**
     * @brief Cut surface
     * 
     * @return int 
     */
    int cut_surface();

    /**
     * @brief Set the tri coplanar object
     * 
     * @param id 
     * @param axis 
     * @param p 
     * @return int 
     */
    int set_tri_coplanar(size_t id, size_t axis, size_t p);

    /**
     * @brief Determine whether the normal of the triangle is positive
     * 
     * @param axis_projection 
     * @param id_tri 
     * @return true 
     * @return false 
     */
    bool is_triangle_normal_positive(size_t axis_projection, size_t id_tri);

    /**
     * @brief Get the axis grid object
     * 
     * @param axis_itr 
     * @param sequence 
     * @param grid_axis 
     * @return size_t 
     */
    size_t get_axis_grid(
        size_t                     axis_itr,
        const std::vector<size_t>& sequence,
        const std::vector<double>& grid_axis);

private:
    /**
     * @brief Get the line vert axis grid id object
     * 
     * @param axis_itr 
     * @param ptr_v 
     * @param grid_axis 
     * @param grid_id 
     * @return size_t 
     */
    size_t get_line_vert_axis_grid_id(
        size_t                     axis_itr,
        const Vert*                ptr_v,
        const std::vector<double>& grid_axis,
        std::vector<size_t>&       grid_id);

    /**
     * @brief Get the line vert grid id object
     * 
     * @param axis_itr 
     * @param p 
     * @param id_tri 
     * @param grid_axis 
     * @param grid_id 
     * @return size_t 
     */
    size_t get_line_vert_grid_id(
        size_t                     axis_itr,
        double*                    p,
        size_t                     id_tri,
        const std::vector<double>& grid_axis,
        std::vector<size_t>&       grid_id);

    /**
     * @brief Get the edge vert axis grid id object
     * 
     * @param axis_itr 
     * @param id_v 
     * @param grid_axis 
     * @param grid_id 
     * @return size_t 
     */
    size_t get_edge_vert_axis_grid_id(
        size_t                     axis_itr,
        size_t                     id_v,
        const std::vector<double>& grid_axis,
        std::vector<size_t>&       grid_id);

    /**
     * @brief Set the projection edge vert axis grid id object
     * 
     * @param axis_projection 
     * @param id_v 
     * @param axis 
     * @param grid_axis 
     * @param id_grid_line 
     * @param grid_id 
     * @param v_projection 
     * @return size_t 
     */
    size_t set_projection_edge_vert_axis_grid_id(
        size_t                     axis_projection,
        size_t                     id_v,
        size_t                     axis,
        const std::vector<double>& grid_axis,
        size_t                     id_grid_line,
        std::vector<size_t>&       grid_id,
        const Eigen::Vector2d      v_projection[2]);

private:
    /**
     * @brief Cut the mesh on the axis
     * 
     * @param axis 
     * @param p 
     * @return std::vector<Ring> 
     */
    std::vector<Ring> cut_mesh_on_axis_at_p(size_t axis, size_t p);

    /**
     * @brief Cut the edge
     * 
     * @param id_tri 
     * @param edge 
     * @return int 
     */
    int cut_edge(size_t id_tri, Edge& edge);

    /**
     * @brief Cut the mesh on the axis
     * 
     * @param id_tri 
     * @param axis 
     * @param p 
     * @return std::vector<size_t> 
     */
    std::vector<size_t> cut_triangle_on_axis_at_p(
        size_t id_tri,
        size_t axis,
        size_t p);

    /**
     * @brief Cut one triangle
     * 
     * @param lower_t 
     * @param upper_t 
     * @param axis 
     * @param p 
     * @param p_grid 
     * @param id_tri 
     * @param ring_lower 
     * @param ring_upper 
     * @param is_coplanar 
     * @return int 
     */
    int cut_one_tri(
        double lower_t,
        double upper_t,
        size_t axis,
        size_t p,
        double p_grid,
        size_t id_tri,
        Ring&  ring_lower,
        Ring&  ring_upper,
        bool&  is_coplanar);

private:
    std::vector<Vert*>     table_vert_ptr_;
    std::vector<Vector3st> table_tri_;
    std::vector<PatchType> table_tri_type_;
    std::vector<size_t>    table_tri_group_id_;

    std::vector<Triangle> table_triangle_;

    Eigen::MatrixXd aabb_;
    size_t          num_span_;
    Eigen::MatrixXd grid_line_;

    std::unordered_map<Vector3st, std::list<std::vector<size_t>>, HashFucLatticeId, IdEqualLattice>            polygon_cell_;
    std::unordered_map<Vector3st, std::vector<std::pair<PatchType, size_t>>, HashFucLatticeId, IdEqualLattice> patch_cell_;
};
