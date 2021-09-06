#pragma once

#include <Eigen/Core>
#include <vector>
#include <array>

/**
 * @brief FEM Geometry Node
 * 
 * @tparam T 
 */
template <typename T>
struct Node
{
    /**
     * @brief Construct a new Node object
     * 
     */
    Node()
    {
        node_left  = nullptr;
        node_right = nullptr;
    }

    Eigen::Matrix<T, 3, 2> aabb;
    std::vector<size_t>    table_tri;
    Node<T>*               node_left;
    Node<T>*               node_right;
};

/**
 * @brief FEM Geometry KdTree
 * 
 * @tparam T 
 */
template <typename T>
struct KdTree
{
    /**
     * @brief Construct a new Kd Tree object
     * 
     */
    KdTree()
    {
        root = nullptr;
    }
    /**
     * @brief Destroy the Kd Tree object
     * 
     */
    ~KdTree();

    /**
     * @brief build the tree
     * 
     * @param path 
     * @return int 
     */
    int build_tree(const char* const path);

    /**
     * @brief Set the up tree object
     * 
     * @param table_t 
     * @param axis 
     * @return Node<T>* 
     */
    Node<T>* setup_tree(const std::vector<size_t>& table_t, size_t axis);

    /**
     * @brief Get the tri soup object
     * 
     * @param path 
     * @return int 
     */
    int get_tri_soup(const char* const path);

    /**
     * @brief destory the tree
     * 
     * @param ptr 
     * @return int 
     */
    int destory_tree(Node<T>* ptr);

    /**
     * @brief Get the tri num object
     * 
     * @return size_t 
     */
    size_t get_tri_num() const;

    /**
     * @brief Get the tri object
     * 
     * @param id_tri 
     * @return Eigen::Matrix<T, 3, 3> 
     */
    Eigen::Matrix<T, 3, 3> get_tri(size_t id_tri) const;

    /**
     * @brief Get the neigh tri object
     * 
     * @param id_tri 
     * @return std::vector<Eigen::Matrix<T, 3, 3>> 
     */
    std::vector<Eigen::Matrix<T, 3, 3>> get_neigh_tri(size_t id_tri) const;

    /**
     * @brief Get the neigh tri object
     * 
     * @param aabb 
     * @param ptr 
     * @param table_neigh_tri 
     * @return int 
     */
    int get_neigh_tri(const Eigen::Matrix<T, 3, 2>& aabb, const Node<T>* const ptr, std::vector<size_t>& table_neigh_tri) const;

    /**
     * @brief Get the aabb object
     * 
     * @param table_tri 
     * @return Eigen::Matrix<T, 3, 2> 
     */
    Eigen::Matrix<T, 3, 2> get_aabb(
        const std::vector<size_t>& table_tri) const;

    /**
     * @brief Get the tri center object
     * 
     * @param table_t 
     * @return std::vector<Eigen::Matrix<T, 3, 1>> 
     */
    std::vector<Eigen::Matrix<T, 3, 1>> get_tri_center(const std::vector<size_t>& table_t) const;

    Node<T>*                                 root;
    std::vector<Eigen::Matrix<T, 3, 1>>      table_vert;
    std::vector<Eigen::Matrix<size_t, 3, 1>> table_tri;
};

/**
 * @brief Determine whether the grid intersects itself
 * 
 * @tparam T 
 * @param path 
 * @return true 
 * @return false 
 */
template <typename T>
bool is_mesh_self_intersection(const char* const path);

/**
 * @brief Determine whether the two triangles intersects itself
 * 
 * @tparam T 
 * @param tri_a 
 * @param tri_b 
 * @return true 
 * @return false 
 */
template <typename T>
bool is_triangle_triangle_self_intersect(const Eigen::Matrix<T, 3, 3>& tri_a, const Eigen::Matrix<T, 3, 3>& tri_b);

/**
 * @brief Determine whether the triangle is degenerate
 * 
 * @tparam T 
 * @param tri 
 * @return true 
 * @return false 
 */
template <typename T>
bool is_triangle_degenerate(const Eigen::Matrix<T, 3, 3>& tri);

/**
 * @brief Determine whether it is a universal connection
 * 
 * @param position_a2b 
 * @return true 
 * @return false 
 */
bool is_universal_connect(const std::array<int, 3>& position_a2b);

/**
 * @brief Determine whether it is a common connection
 * 
 * @tparam T 
 * @param tri_a 
 * @param tri_b 
 * @param position_a2b 
 * @param position_b2a 
 * @return int 
 */
template <typename T>
int is_common_condition(const Eigen::Matrix<T, 3, 3>& tri_a, const Eigen::Matrix<T, 3, 3>& tri_b, const std::array<int, 3>& position_a2b, const std::array<int, 3>& position_b2a);

/**
 * @brief Determine whether the triangle is above the triangle
 * 
 * @param position 
 * @return true 
 * @return false 
 */
bool is_triangle_above_triangle(const std::array<int, 3>& position);

/**
 * @brief Determine whether two triangles intersect
 * 
 * @tparam T 
 * @param tri_a 
 * @param tri_b 
 * @return true 
 * @return false 
 */
template <typename T>
bool is_triangle_to_triangle_intersect(const Eigen::Matrix<T, 3, 3>& tri_a, const Eigen::Matrix<T, 3, 3>& tri_b);

/**
 * @brief Determine whether the triangles intersects themself
 * 
 * @tparam T 
 * @param tri_a 
 * @param tri_b 
 * @param position_a2b 
 * @param position_b2a 
 * @return true 
 * @return false 
 */
template <typename T>
bool is_triangle_triangle_self_intersect(const Eigen::Matrix<T, 3, 3>& tri_a, const Eigen::Matrix<T, 3, 3>& tri_b, const std::array<int, 3>& position_a2b, const std::array<int, 3>& position_b2a);

/**
 * @brief Determine if the triangles intersect or are coplanar
 * 
 * @tparam T 
 * @param tri_a 
 * @param tri_b 
 * @return true 
 * @return false 
 */
template <typename T>
bool is_coplanar_triangle_triangle_intersect(const Eigen::Matrix<T, 3, 3>& tri_a, const Eigen::Matrix<T, 3, 3>& tri_b);

/**
 * @brief Determine whether the vertex intersects the triangle
 * 
 * @tparam T 
 * @param v 
 * @param p 
 * @param tri 
 * @param coplanar_v_triangle 
 * @return true 
 * @return false 
 */
template <typename T>
bool is_vert_triangle_intersect(const Eigen::Matrix<T, 3, 1>& v, const Eigen::Matrix<T, 3, 1>& p, const Eigen::Matrix<T, 3, 3>& tri, bool coplanar_v_triangle = true);

/**
 * @brief Determine if the edges intersect or are coplanar
 * 
 * @tparam T 
 * @param e 
 * @param p 
 * @param tri 
 * @return true 
 * @return false 
 */
template <typename T>
bool is_coplanar_edge_triangle_intersect(const Eigen::Matrix<T, 3, 2>& e, const Eigen::Matrix<T, 3, 1>& p, const Eigen::Matrix<T, 3, 3>& tri);

/**
 * @brief Determine whether the edge intersects the triangle
 * 
 * @tparam T 
 * @param e 
 * @param tri 
 * @return true 
 * @return false 
 */
template <typename T>
bool is_edge_triangle_intersect(const Eigen::Matrix<T, 3, 2>& e, const Eigen::Matrix<T, 3, 3>& tri);

/**
 * @brief Determine whether the edges intersect
 * 
 * @tparam T 
 * @param e 
 * @param et 
 * @param p 
 * @return true 
 * @return false 
 */
template <typename T>
bool is_edge_edge_intersect(const Eigen::Matrix<T, 3, 2>& e, const Eigen::Matrix<T, 3, 2>& et, const Eigen::Matrix<T, 3, 1>& p);
