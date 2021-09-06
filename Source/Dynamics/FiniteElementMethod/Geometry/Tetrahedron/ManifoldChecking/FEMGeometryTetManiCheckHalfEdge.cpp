#include "FEMGeometryTetManiCheckHalfEdge.h"
#include <fstream>
#include <unordered_map>
#include <iostream>
#include <string>
#include <map>
#include <array>
#include "FEMGeometryTetManiCheckHashKey.h"

using namespace std;
using namespace Eigen;

typedef Eigen::Matrix<size_t, 3, 1> Vector3st;
typedef Eigen::Matrix<size_t, 2, 1> Vector2st;

typedef std::function<size_t(const Eigen::Vector3d&)>                       Vector3dTripleFunc;
typedef std::function<bool(const Eigen::Vector3d&, const Eigen::Vector3d&)> Vector3dTwoTripleFunc;
typedef std::function<size_t(const Vector2st&)>                             Vector2dPairFunc;
typedef std::function<bool(const Vector2st&, const Vector2st&)>             Vector2dTwoPairFunc;

HalfEdgeMesh::HalfEdgeMesh()
{
}

HalfEdgeMesh::HalfEdgeMesh(const char* const path)
{
    string str_path(path);
    if (str_path.find(".") == std::string::npos
        || str_path.substr(str_path.rfind(".")) != ".obj")
    {
        cerr << "only obj format supported" << endl;
        return;
    }

    ifstream f_in(path);
    if (!f_in)
    {
        cerr << "error in file open" << endl;
        return;
    }

    Vector3dTripleFunc    HashFunc3 = bind(HashFunc<Vector3d, 3>, std::placeholders::_1);
    Vector3dTwoTripleFunc EqualKey3 = bind(EqualKey<Vector3d, 3>, std::placeholders::_1, std::placeholders::_2);

    unordered_map<Vector3d, size_t, Vector3dTripleFunc, Vector3dTwoTripleFunc> map_vert(1, HashFunc3, EqualKey3);
    map<size_t, size_t>                                                        map_old_to_new_vert_id;

    Vector2dPairFunc    HashFunc2 = bind(HashFunc<Vector2st, 2>, std::placeholders::_1);
    Vector2dTwoPairFunc EqualKey2 = bind(EqualKey<Vector2st, 2>, std::placeholders::_1, std::placeholders::_2);

    unordered_map<Vector2st, size_t, Vector2dPairFunc, Vector2dTwoPairFunc> map_edge(1, HashFunc2, EqualKey2);
    vector<Vector2st>                                                       vec_ve;

    string str_line;
    while (getline(f_in, str_line))
    {
        if (str_line.size() < 2)
            continue;

        if (str_line[0] == 'v'
            && str_line[1] == ' ')
        {
            Vector3d v;
            sscanf(str_line.c_str(),
                   "%*s %lf %lf %lf",
                   &v(0),
                   &v(1),
                   &v(2));
            if (!map_vert.count(v))
            {
                map_vert.emplace(v, map_vert.size());
                map_old_to_new_vert_id.emplace(
                    map_old_to_new_vert_id.size() + 1, map_vert.size() - 1);
                vert_data_.push_back(VertData(v));
                HalfVert he_v;
                he_v.v = vert_data_.size() - 1;
                he_v_.push_back(he_v);
            }
            else
            {
                map_old_to_new_vert_id.emplace(
                    map_old_to_new_vert_id.size() + 1, map_vert.at(v));
            }
            continue;
        }
        else if (str_line[0] == 'f'
                 && str_line[1] == ' ')
        {
            Vector3st tri_vert;
            if (str_line.find("/") == string::npos)
                sscanf(str_line.c_str(),
                       "%*s %zu %zu %zu",
                       &tri_vert(0),
                       &tri_vert(1),
                       &tri_vert(2));
            else
                sscanf(str_line.c_str(),
                       "%*s %zu%*s %zu%*s %zu%*s",
                       &tri_vert(0),
                       &tri_vert(1),
                       &tri_vert(2));

            for (size_t axis = 0; axis < 3; ++axis)
            {
                assert(map_old_to_new_vert_id.count(tri_vert(axis)));
                tri_vert(axis) = map_old_to_new_vert_id.at(tri_vert(axis));
            }
            if (tri_vert[0] == tri_vert[1] || tri_vert[1] == tri_vert[2]
                || tri_vert[2] == tri_vert[0])
            {
                cerr << "[ warning ]: degenerate triangle appear" << endl;
                continue;
            }
            array<HalfEdge, 3> he_e;
            HalfFace           he_f;
            he_f.he_e = he_e_.size();
            for (size_t i = 0; i < 3; ++i)
            {
                he_v_[tri_vert[i]].he_e = he_e_.size() + i;
                he_e[i].he_f            = he_f_.size();
                he_e[i].he_v            = tri_vert[(i + 1) % 3];
                he_e[i].next_e          = he_e_.size() + (i + 1) % 3;

                Vector2st v_e(tri_vert[i], tri_vert[(i + 1) % 3]);
                map_edge.emplace(v_e, he_e_.size() + i);
                vec_ve.push_back(v_e);
            }
            he_e_.insert(he_e_.end(), he_e.begin(), he_e.end());
            he_f_.push_back(he_f);
            continue;
        }
    }
    f_in.close();

    const size_t num_e = he_e_.size();
    for (size_t i = 0; i < num_e; ++i)
    {
        const auto& e = vec_ve[i];
        Vector2st   pair_ve(e[1], e[0]);
        if (map_edge.count(pair_ve) == 1)
            he_e_[i].pair_e = map_edge[pair_ve];
        else
            cerr << "This is not a close mesh" << endl;
    }

    face_group_ = UnionFindSet(he_f_.size());
    set_face_group();
}

size_t HalfEdgeMesh::get_face_num() const
{
    return he_f_.size();
}

size_t HalfEdgeMesh::get_vert_num() const
{
    return he_v_.size();
}

std::vector<size_t> HalfEdgeMesh::get_vert_neigh_face(size_t id_v) const
{
    list<size_t> vec_f;
    const size_t id_e = he_v_.at(id_v).he_e;
    vec_f.push_back(he_e_.at(id_e).he_f);

    if (he_e_.at(id_e).pair_e != numeric_limits<size_t>::max())
    {
        size_t id_next_e = he_e_.at(he_e_.at(id_e).pair_e).next_e;
        int    f_count   = 0;
        while (id_next_e != id_e)
        {
            vec_f.push_back(he_e_.at(id_next_e).he_f);
            if (he_e_.at(id_next_e).pair_e != numeric_limits<size_t>::max())
                id_next_e = he_e_.at(he_e_.at(id_next_e).pair_e).next_e;
            else
            {
                get_vert_front_neigh_face(id_e, vec_f);
                break;
            }
            ++f_count;
            if (f_count > 200)
            {
                cerr << "more than 200 faces around vertx in end neighbor,,," << endl;
            }
        }
    }
    else
    {
        get_vert_front_neigh_face(id_e, vec_f);
    }

    return vector<size_t>(vec_f.begin(), vec_f.end());
}

int HalfEdgeMesh::get_vert_front_neigh_face(size_t id_e, std::list<size_t>& vec_f) const
{
    size_t prev_e = he_e_.at(he_e_.at(id_e).next_e).next_e;
    if (he_e_.at(prev_e).pair_e != numeric_limits<size_t>::max())
    {
        size_t pair_e      = he_e_.at(prev_e).pair_e;
        size_t next_prev_e = he_e_.at(he_e_.at(pair_e).next_e).next_e;
        int    f_count     = 0;
        while (next_prev_e != prev_e)
        {
            vec_f.push_front(he_e_.at(next_prev_e).he_f);
            if (he_e_.at(next_prev_e).pair_e != numeric_limits<size_t>::max())
            {
                pair_e      = he_e_.at(next_prev_e).pair_e;
                next_prev_e = he_e_.at(he_e_.at(pair_e).next_e).next_e;
            }
            else
                break;

            ++f_count;
            if (f_count > 200)
            {
                cerr << "more than 200 face around a vertex in front neighbor" << endl;
                return -1;
            }
        }
    }

    return 0;
}

Eigen::Matrix3d HalfEdgeMesh::get_tri(size_t id_f) const
{
    array<size_t, 3> tri_vert = get_tri_vert_id(id_f);

    Matrix3d tri = Matrix3d::Constant(0);
    for (size_t i = 0; i < 3; ++i)
    {
        tri.col(i) = vert_data_.at(tri_vert[i]).v;
    }

    return tri;
}

const Eigen::Vector3d& HalfEdgeMesh::get_vert(size_t id_v) const
{
    return vert_data_.at(id_v).v;
}

std::vector<size_t> HalfEdgeMesh::get_face_neigh_face(size_t id_f) const
{
    assert(id_f < he_f_.size());
    const HalfFace&  he_f = he_f_.at(id_f);
    array<size_t, 3> e_f;
    e_f[0] = he_f.he_e;
    e_f[1] = he_e_.at(e_f[0]).next_e;
    e_f[2] = he_e_.at(e_f[1]).next_e;

    vector<size_t> nf(3);
    for (size_t itr = 0; itr < 3; ++itr)
    {
        size_t pe = he_e_.at(e_f[itr]).pair_e;
        nf[itr]   = he_e_.at(pe).he_f;
    }

    return nf;
}

std::array<size_t, 3> HalfEdgeMesh::get_tri_vert_id(size_t id_f) const
{
    array<size_t, 3> face_he = get_tri_edge(id_f);
    array<size_t, 3> vert_id;
    for (size_t i = 0; i < 3; ++i)
    {
        size_t prev_e = he_e_.at(he_e_.at(face_he[i]).next_e).next_e;
        vert_id[i]    = he_e_.at(prev_e).he_v;
    }

    return vert_id;
}

size_t HalfEdgeMesh::get_edge_num() const
{
    return he_e_.size();
}

std::array<size_t, 2> HalfEdgeMesh::get_edge_neighbor_face(size_t edge_id) const
{
    array<size_t, 2> edge_face;
    edge_face[0] = he_e_.at(edge_id).he_f;
    edge_face[1] = he_e_.at(he_e_.at(edge_id).pair_e).he_f;

    return edge_face;
}

std::array<size_t, 2> HalfEdgeMesh::get_edge_vert_id(size_t edge_id) const
{
    array<size_t, 2> edge_vert;
    edge_vert[0] = he_e_.at(edge_id).he_v;
    edge_vert[1] = he_e_.at(he_e_.at(edge_id).pair_e).he_v;
    swap(edge_vert[0], edge_vert[1]);

    return edge_vert;
}

std::array<Eigen::Vector3d, 2> HalfEdgeMesh::get_edge(size_t edge_id) const
{
    array<size_t, 2>   edge_vert_id = get_edge_vert_id(edge_id);
    array<Vector3d, 2> edge_vert;
    for (size_t i = 0; i < 2; ++i)
        edge_vert[i] = vert_data_.at(edge_vert_id[i]).v;

    return edge_vert;
}

Eigen::MatrixXd HalfEdgeMesh::get_aabb() const
{
    MatrixXd aabb(3, 2);
    aabb.col(0) = Vector3d(numeric_limits<double>::max(),
                           numeric_limits<double>::max(),
                           numeric_limits<double>::max());
    aabb.col(1) = Vector3d(-numeric_limits<double>::max(),
                           -numeric_limits<double>::max(),
                           -numeric_limits<double>::max());

    for (const auto& ptr : vert_data_)
    {
        const Vector3d& vert = ptr.v;
        for (size_t i = 0; i < 3; ++i)
        {
            if (vert(i) < aabb(i, 0))
                aabb(i, 0) = vert(i);
            if (vert(i) > aabb(i, 1))
                aabb(i, 1) = vert(i);
        }
    }

    return aabb;
}

std::vector<size_t> HalfEdgeMesh::get_unordered_edge_id() const
{
    ArrayPairFunc                                                            UnorderedHashFuncPair = bind(UnorderedHashFunc<array<size_t, 2>, 2>, std::placeholders::_1);
    ArrayTwoPairFunc                                                         UnorderedEqualKeyPair = bind(UnorderedEqualKey<array<size_t, 2>, 2>, std::placeholders::_1, std::placeholders::_2);
    unordered_map<array<size_t, 2>, size_t, ArrayPairFunc, ArrayTwoPairFunc> edge_id(1, UnorderedHashFuncPair, UnorderedEqualKeyPair);

    vector<size_t> unordered_edge_id;
    const size_t   half_edge_edge_num = he_e_.size();
    for (size_t i = 0; i < half_edge_edge_num; ++i)
    {
        array<size_t, 2> edge_vert = get_edge_vert_id(i);
        if (edge_id.count(edge_vert))
            continue;
        unordered_edge_id.push_back(i);
        edge_id.emplace(edge_vert, i);
    }

    return unordered_edge_id;
}

std::array<size_t, 3> HalfEdgeMesh::get_tri_edge(size_t face_id) const
{
    size_t           he           = he_f_.at(face_id).he_e;
    size_t           next_he      = he_e_.at(he).next_e;
    size_t           next_next_he = he_e_.at(next_he).next_e;
    array<size_t, 3> face_he      = { he, next_he, next_next_he };
    return face_he;
}

int HalfEdgeMesh::set_face_group()
{
    size_t face_num = he_f_.size();
    for (size_t i = 0; i < face_num; ++i)
    {
        auto neigh_face = get_face_neigh_face(i);
        for (auto f : neigh_face)
        {
            set_face_connect(i, f);
        }
    }

    return 0;
}

void HalfEdgeMesh::set_face_connect(size_t f1, size_t f2)
{
    face_group_.set_union(f1, f2);
}

bool HalfEdgeMesh::is_face_connect(size_t f1, size_t f2) const
{
    return face_group_.is_connected(f1, f2);
}

std::unordered_map<size_t, std::vector<size_t>> HalfEdgeMesh::get_face_group() const
{
    return face_group_.get_group();
}
