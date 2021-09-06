#pragma once

#include <Eigen/Core>
#include <vector>

typedef Eigen::Matrix<size_t, 3, 1> Vector3st;

class Mesh;
class Loop;

/**
 * @brief FEM Geometry Vert
 * 
 */
class Vert
{
public:
    /**
     * @brief Construct a new Vert object
     * 
     * @param v 
     */
    Vert(const Eigen::Vector3d& v);

    /**
     * @brief Construct a new Vert object
     * 
     */
    Vert();

    /**
     * @brief Get the vert coord object
     * 
     * @return const Eigen::Vector3d 
     */
    virtual const Eigen::Vector3d get_vert_coord() const;

    /**
     * @brief Initialize the vert object
     * 
     * @param mesh 
     * @return int 
     */
    static int                    init_vert(Mesh* mesh);

    /**
     * @brief Set the grid line object
     * 
     * @param axis 
     * @param grid 
     * @return int 
     */
    int                           set_grid_line(size_t axis, size_t grid);

    /**
     * @brief Get the grid line object
     * 
     * @param axis 
     * @return size_t 
     */
    size_t                        get_grid_line(size_t axis) const;

    /**
     * @brief Get the vert id object
     * 
     * @return size_t 
     */
    size_t                        get_vert_id() const;

    /**
     * @brief Set the id object
     * 
     * @param id 
     * @return int 
     */
    int                           set_id(size_t id);

    /**
     * @brief Determine whether the vertex is on axis grid
     * 
     * @param axis 
     * @param id_grid 
     * @return true 
     * @return false 
     */
    bool                          is_vert_on_axis_grid(size_t axis, size_t id_grid) const;

    /**
     * @brief Get the grid id object
     * 
     * @return const Vector3st& 
     */
    const Vector3st&              get_grid_id() const;

    /**
     * @brief Determine whether the vertexs is the same
     * 
     * @param id_v 
     * @return true 
     * @return false 
     */
    virtual bool is_same_vert_with(size_t id_v) const;

    /**
     * @brief Get the edge vert info object
     * 
     * @param id_v1 
     * @param id_v2 
     * @param axis 
     * @param id_grid_line 
     * @return int 
     */
    virtual int  get_edge_vert_info(size_t& id_v1, size_t& id_v2, size_t& axis, size_t& id_grid_line) const;

    /**
     * @brief Get the line vert info object
     * 
     * @param a1 
     * @param a2 
     * @param g1 
     * @param g2 
     * @param id_tri 
     * @return int 
     */
    virtual int  get_line_vert_info(
         size_t& a1,
         size_t& a2,
         size_t& g1,
         size_t& g2,
         size_t& id_tri) const;

    /**
     * @brief Determine whether the vertex is a triangle vertex
     * 
     * @return true 
     * @return false 
     */
    virtual bool is_triangle_vert() const;

    /**
     * @brief Determine whether the vertex is a grid vertex
     * 
     * @return true 
     * @return false 
     */
    virtual bool is_grid_vert() const;

    /**
     * @brief Determine whether the vertex is a edge vertex
     * 
     * @return true 
     * @return false 
     */
    virtual bool is_edge_vert() const;

    /**
     * @brief Determine whether the vertex is a line vertex
     * 
     * @return true 
     * @return false 
     */
    virtual bool is_line_vert() const;

    /**
     * @brief Determine whether the vertex is on a triangle edge
     * 
     * @return true 
     * @return false 
     */
    bool is_on_triangle_edge() const;

    /**
     * @brief Set the vert on triangle edge object
     * 
     * @return int 
     */
    int  set_vert_on_triangle_edge();

    /**
     * @brief Determine whether on a triangle edge
     * 
     * @return true 
     * @return false 
     */
    bool is_on_triangle_vert() const;

    /**
     * @brief Set the vert on triangle vert object
     * 
     * @return int 
     */
    int  set_vert_on_triangle_vert();

    /**
     * @brief Get the triangle id object
     * 
     * @return std::vector<size_t> 
     */
    virtual std::vector<size_t> get_triangle_id() const;

    /**
     * @brief Set the edge v object
     * 
     * @param v1_e 
     * @param v2_e 
     * @return int 
     */
    int                         set_edge_v(size_t v1_e, size_t v2_e);

    /**
     * @brief Get the edge v object
     * 
     * @param v1_e 
     * @param v2_e 
     * @return int 
     */
    int                         get_edge_v(size_t& v1_e, size_t& v2_e) const;

    /**
     * @brief Get the edge vert object
     * 
     * @param v1 
     * @param v2 
     * @return int 
     */
    virtual int get_edge_vert(size_t& v1, size_t& v2) const;

protected:
    Eigen::Vector3d v_;
    Vector3st       grid_line_;
    size_t          id_;
    bool            is_on_triangle_edge_;
    bool            is_on_triangle_vert_;
    size_t          id_edge_v_[2];

    // protected:
public:
    static Mesh* mesh_;
};

/**
 * @brief FEM Geometry VertEdge
 * 
 */
class VertEdge : public Vert  // vert on triangle edge
{
public:
    /**
     * @brief Construct a new Vert Edge object
     * 
     * @param id_v1 
     * @param id_v2 
     * @param axis 
     * @param id_grid_line 
     */
    VertEdge(size_t id_v1, size_t id_v2, size_t axis, size_t id_grid_line);

    /**
     * @brief Get the vert coord object
     * 
     * @return const Eigen::Vector3d 
     */
    virtual const Eigen::Vector3d get_vert_coord() const;

    /**
     * @brief determine whether the vertexs is same
     * 
     * @param id_v 
     * @return true 
     * @return false 
     */
    virtual bool                  is_same_vert_with(size_t id_v) const;

    /**
     * @brief Get the edge vert info object
     * 
     * @param id_v1 
     * @param id_v2 
     * @param axis 
     * @param id_grid_line 
     * @return int 
     */
    virtual int                   get_edge_vert_info(size_t& id_v1, size_t& id_v2, size_t& axis, size_t& id_grid_line) const;

    /**
     * @brief Get the triangle id object
     * 
     * @return std::vector<size_t> 
     */
    virtual std::vector<size_t> get_triangle_id() const;

    /**
     * @brief Get the edge vert object
     * 
     * @param v1 
     * @param v2 
     * @return int 
     */
    virtual int                 get_edge_vert(size_t& v1, size_t& v2) const;

    /**
     * @brief Determine whether the vertex is on a triangle edge
     * 
     * @return true 
     * @return false 
     */
    virtual bool                is_triangle_vert() const;

    /**
     * @brief Determine whether the vertex is on a triangle edge
     * 
     * @return true 
     * @return false 
     */
    virtual bool                is_edge_vert() const;

private:
    size_t id_v_[2];
    size_t axis_;
    size_t id_grid_;
};

class VertLine : public Vert  // vert on cutted edge line
{
public:
    /**
     * @brief Construct a new Vert Line object
     * 
     * @param id_tri 
     * @param axis_1 
     * @param id_grid_1 
     * @param axis_2 
     * @param id_grid_2 
     */
    VertLine(size_t id_tri, size_t axis_1, size_t id_grid_1, size_t axis_2, size_t id_grid_2);

    /**
     * @brief Get the vert coord object
     * 
     * @return const Eigen::Vector3d 
     */
    virtual const Eigen::Vector3d get_vert_coord() const;

    /**
     * @brief Determine whether the vertexs is same
     * 
     * @param id_v 
     * @return true 
     * @return false 
     */
    virtual bool                  is_same_vert_with(size_t id_v) const;

    /**
     * @brief Get the line vert info object
     * 
     * @param a1 
     * @param a2 
     * @param g1 
     * @param g2 
     * @param id_tri 
     * @return int 
     */
    virtual int                   get_line_vert_info(
                          size_t& a1,
                          size_t& a2,
                          size_t& g1,
                          size_t& g2,
                          size_t& id_tri) const;

    /**
     * @brief Determine whether the object is a triangle vertex
     * 
     * @return true 
     * @return false 
     */
    virtual bool is_triangle_vert() const;

    /**
     * @brief Determine whether the object is a line vertex
     * 
     * @return true 
     * @return false 
     */
    virtual bool is_line_vert() const;

private:
    size_t axis_[2];
    size_t id_grid_[2];
    size_t id_tri_;
};

/**
 * @brief FEM Geometry VertGrid
 * 
 */
class VertGrid : public Vert
{
public:
    /**
     * @brief Construct a new Vert Grid object
     * 
     * @param axis 
     * @param id_grid_1 
     * @param axis_cut 
     * @param id_grid_2 
     * @param axis_projection 
     * @param id_grid_3 
     */
    VertGrid(size_t axis, size_t id_grid_1, size_t axis_cut, size_t id_grid_2, size_t axis_projection, size_t id_grid_3);

public:
    /**
     * @brief Determine whether the vertexs is same
     * 
     * @param id_v 
     * @return true 
     * @return false 
     */
    virtual bool                  is_same_vert_with(size_t id_v) const;

    /**
     * @brief Get the vert coord object
     * 
     * @return const Eigen::Vector3d 
     */
    virtual const Eigen::Vector3d get_vert_coord() const;

    /**
     * @brief Determine whether the object is a triangle vertex
     * 
     * @return true 
     * @return false 
     */
    virtual bool                  is_triangle_vert() const;

    /**
     * @brief Determine whether the object is a grid vertex
     * 
     * @return true 
     * @return false 
     */
    virtual bool                  is_grid_vert() const;
};
