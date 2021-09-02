#ifndef HALF_EDGE_JJ_H
#define HALF_EDGE_JJ_H

#include <Eigen/Core>
#include <vector>
#include <limits>
#include <list>
#include <unordered_map>
#include "union_find_set.h"

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

struct HalfFace
{
    HalfFace()
    {
        he_e = std::numeric_limits<size_t>::max();
    }

    size_t he_e;
};

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

class HalfEdgeMesh
{
public:
    HalfEdgeMesh();
    HalfEdgeMesh(const char* const path);

public:
    size_t get_face_num() const;
    size_t get_vert_num() const;
    size_t get_edge_num() const;

    std::vector<size_t>            get_vert_neigh_face(size_t id_v) const;
    std::vector<size_t>            get_face_neigh_face(size_t id_f) const;
    Eigen::Matrix3d                get_tri(size_t id_f) const;
    std::array<size_t, 3>          get_tri_edge(size_t face_id) const;
    std::array<size_t, 3>          get_tri_vert_id(size_t id_f) const;
    const Eigen::Vector3d&         get_vert(size_t id_v) const;
    std::array<size_t, 2>          get_edge_neighbor_face(size_t edge_id) const;
    std::array<size_t, 2>          get_edge_vert_id(size_t edge_id) const;
    std::array<Eigen::Vector3d, 2> get_edge(size_t edge_id) const;
    Eigen::MatrixXd                get_aabb() const;
    //unordered edge which half edge is derivated from
    std::vector<size_t>                             get_unordered_edge_id() const;
    void                                            set_face_connect(size_t f1, size_t f2);
    bool                                            is_face_connect(size_t f1, size_t f2) const;
    std::unordered_map<size_t, std::vector<size_t>> get_face_group() const;

private:
    int get_vert_front_neigh_face(size_t id_e, std::list<size_t>& vec_f) const;
    int set_face_group();

private:
    std::vector<HalfVert> he_v_;
    std::vector<HalfEdge> he_e_;
    std::vector<HalfFace> he_f_;
    std::vector<VertData> vert_data_;

    UnionFindSet face_group_;
};

#endif  // HALF_EDGE_JJ_H
