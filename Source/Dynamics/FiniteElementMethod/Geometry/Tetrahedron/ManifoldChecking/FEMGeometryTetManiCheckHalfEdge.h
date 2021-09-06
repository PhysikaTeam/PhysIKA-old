#pragma once

#include <Eigen/Core>
#include <vector>
#include <limits>
#include <list>
#include <unordered_map>
#include "FEMGeometryTetManiCheckUnionFindSet.h"

/**
 * @brief FEM Geometry HalfVert
 * 
 */
struct HalfVert
{
    HalfVert()
    {
        he_e = std::numeric_limits<size_t>::max();
        v    = std::numeric_limits<size_t>::max();
    }

    size_t he_e;
    size_t v;
};

/**
 * @brief FEM Geometry HalfEdge
 * 
 */
struct HalfEdge
{
    HalfEdge()
    {
        pair_e = std::numeric_limits<size_t>::max();
        next_e = std::numeric_limits<size_t>::max();
        he_f   = std::numeric_limits<size_t>::max();
        he_v   = std::numeric_limits<size_t>::max();
    }
    size_t pair_e;
    size_t next_e;
    size_t he_f;
    size_t he_v;
};

/**
 * @brief FEM Geometry HalfFace
 * 
 */
struct HalfFace
{
    HalfFace()
    {
        he_e = std::numeric_limits<size_t>::max();
    }

    size_t he_e;
};

/**
 * @brief FEM Geometry VertData
 * 
 */
struct VertData
{
    VertData(const Eigen::Vector3d& data)
        : v(data) {}
    VertData()
    {
        v = Eigen::Vector3d::Constant(0);
    }

    Eigen::Vector3d v;
};

/**
 * @brief FEM Geometry HalfEdgeMesh
 * 
 */
class HalfEdgeMesh
{
public:
    /**
     * @brief Construct a new Half Edge Mesh object
     * 
     */
    HalfEdgeMesh();

    /**
     * @brief Construct a new Half Edge Mesh object
     * 
     * @param path 
     */
    HalfEdgeMesh(const char* const path);

public:
    /**
     * @brief Get the face num object
     * 
     * @return size_t 
     */
    size_t get_face_num() const;

    /**
     * @brief Get the vert num object
     * 
     * @return size_t 
     */
    size_t get_vert_num() const;

    /**
     * @brief Get the edge num object
     * 
     * @return size_t 
     */
    size_t get_edge_num() const;

    /**
     * @brief Get the vert neigh face object
     * 
     * @param id_v 
     * @return std::vector<size_t> 
     */
    std::vector<size_t>            get_vert_neigh_face(size_t id_v) const;

    /**
     * @brief Get the face neigh face object
     * 
     * @param id_f 
     * @return std::vector<size_t> 
     */
    std::vector<size_t>            get_face_neigh_face(size_t id_f) const;

    /**
     * @brief Get the tri object
     * 
     * @param id_f 
     * @return Eigen::Matrix3d 
     */
    Eigen::Matrix3d                get_tri(size_t id_f) const;

    /**
     * @brief Get the tri edge object
     * 
     * @param face_id 
     * @return std::array<size_t, 3> 
     */
    std::array<size_t, 3>          get_tri_edge(size_t face_id) const;

    /**
     * @brief Get the tri vert id object
     * 
     * @param id_f 
     * @return std::array<size_t, 3> 
     */
    std::array<size_t, 3>          get_tri_vert_id(size_t id_f) const;

    /**
     * @brief Get the vert object
     * 
     * @param id_v 
     * @return const Eigen::Vector3d& 
     */
    const Eigen::Vector3d&         get_vert(size_t id_v) const;

    /**
     * @brief Get the edge neighbor face object
     * 
     * @param edge_id 
     * @return std::array<size_t, 2> 
     */
    std::array<size_t, 2>          get_edge_neighbor_face(size_t edge_id) const;

    /**
     * @brief Get the edge vert id object
     * 
     * @param edge_id 
     * @return std::array<size_t, 2> 
     */
    std::array<size_t, 2>          get_edge_vert_id(size_t edge_id) const;
    
    /**
     * @brief Get the edge object
     * 
     * @param edge_id 
     * @return std::array<Eigen::Vector3d, 2> 
     */
    std::array<Eigen::Vector3d, 2> get_edge(size_t edge_id) const;

    /**
     * @brief Get the aabb object
     * 
     * @return Eigen::MatrixXd 
     */
    Eigen::MatrixXd                get_aabb() const;
    /**
     * @brief unordered edge which half edge is derivated from
     * 
     * @return std::vector<size_t> 
     */
    std::vector<size_t>                             get_unordered_edge_id() const;

    /**
     * @brief Set the face connect object
     * 
     * @param f1 
     * @param f2 
     */
    void                                            set_face_connect(size_t f1, size_t f2);

    /**
     * @brief determine whether the faces are connected
     * 
     * @param f1 
     * @param f2 
     * @return true 
     * @return false 
     */
    bool                                            is_face_connect(size_t f1, size_t f2) const;

    /**
     * @brief Get the face group object
     * 
     * @return std::unordered_map<size_t, std::vector<size_t>> 
     */
    std::unordered_map<size_t, std::vector<size_t>> get_face_group() const;

private:
    /**
     * @brief Get the vert front neigh face object
     * 
     * @param id_e 
     * @param vec_f 
     * @return int 
     */
    int get_vert_front_neigh_face(size_t id_e, std::list<size_t>& vec_f) const;

    /**
     * @brief Set the face group object
     * 
     * @return int 
     */
    int set_face_group();

private:
    std::vector<HalfVert> he_v_;
    std::vector<HalfEdge> he_e_;
    std::vector<HalfFace> he_f_;
    std::vector<VertData> vert_data_;

    UnionFindSet face_group_;
};
