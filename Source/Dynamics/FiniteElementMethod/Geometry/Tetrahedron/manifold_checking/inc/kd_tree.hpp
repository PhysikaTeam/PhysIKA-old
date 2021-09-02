#ifndef __KD_TREE_HPP__
#define __KD_TREE_HPP__

#include "manifold_check.h"
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <vector>
#include <numeric>
#include <Eigen/Core>
#include <random>
#include "hash_key.h"

using namespace std;
using namespace Eigen;

typedef Eigen::Matrix<size_t, 3, 1> Vector3st;
typedef Eigen::Matrix<size_t, 2, 1> Vector2st;

typedef std::function<size_t(const Eigen::Vector3d&)>                       Vector3dTripleFunc;
typedef std::function<bool(const Eigen::Vector3d&, const Eigen::Vector3d&)> Vector3dTwoTripleFunc;
typedef std::function<size_t(const Vector2st&)>                             Vector2dPairFunc;
typedef std::function<bool(const Vector2st&, const Vector2st&)>             Vector2dTwoPairFunc;

template <typename T>
int KdTree<T>::build_tree(const char* const path)
{
    get_tri_soup(path);
    if (table_tri.empty())
    {
        cerr << "empty model" << endl;
        return 1;
    }
    vector<size_t> table_tri_id(table_tri.size());
    iota(table_tri_id.begin(), table_tri_id.end(), 0);
    constexpr size_t axis = 0;
    root                  = setup_tree(table_tri_id, axis);
    return 0;
}

template <typename T>
Node<T>* KdTree<T>::setup_tree(const std::vector<size_t>& table_tri_id, size_t axis)
{
    Node<T>* ptr   = new Node<T>();
    ptr->table_tri = table_tri_id;
    ptr->aabb      = get_aabb(table_tri_id);
    if (table_tri_id.size() == 1)
        return ptr;

    vector<Matrix<T, 3, 1>> table_c = get_tri_center(table_tri_id);

    vector<size_t> table_index(table_c.size());
    iota(table_index.begin(), table_index.end(), 0);
    sort(table_index.begin(), table_index.end(), [&table_c, axis](size_t l, size_t r) -> bool {
        return table_c[l](axis) < table_c[r](axis);
    });

    size_t         mid = table_c.size() / 2;
    vector<size_t> table_lower_index(table_index.begin(), table_index.begin() + mid);
    vector<size_t> table_upper_index(table_index.begin() + mid, table_index.end());
    vector<size_t> table_lower_tri;
    vector<size_t> table_upper_tri;
    transform(table_lower_index.begin(), table_lower_index.end(), back_inserter(table_lower_tri), [&table_tri_id](size_t i) {
        return table_tri_id.at(i);
    });
    transform(table_upper_index.begin(), table_upper_index.end(), back_inserter(table_upper_tri), [&table_tri_id](size_t i) {
        return table_tri_id.at(i);
    });
    ptr->node_left  = setup_tree(table_lower_tri, (axis + 1) % 3);
    ptr->node_right = setup_tree(table_upper_tri, (axis + 1) % 3);

    return ptr;
}

template <typename T>
std::vector<Eigen::Matrix<T, 3, 1>> KdTree<T>::get_tri_center(const std::vector<size_t>& table_tri_id) const
{
    vector<Matrix<T, 3, 1>> table_c;
    for (auto id_tri : table_tri_id)
    {
        const Matrix<size_t, 3, 1>& tri = table_tri.at(id_tri);
        Matrix<T, 3, 1>             center(0, 0, 0);
        for (size_t i = 0; i < 3; ++i)
        {
            center += table_vert.at(tri[i]);
        }
        center /= 3.0;
        table_c.push_back(center);
    }

    return table_c;
}

template <typename T>
int KdTree<T>::get_tri_soup(const char* const path)
{
    string str_path(path);
    if (str_path.find(".") == std::string::npos
        || str_path.substr(str_path.rfind(".")) != ".obj")
    {
        cerr << "only obj format supported" << endl;
        return -1;
    }

    ifstream f_in(path);
    if (!f_in)
    {
        cerr << "error in file open" << endl;
        return -1;
    }

    Vector3dTripleFunc    HashFunc3 = bind(HashFunc<Vector3d, 3>, std::placeholders::_1);
    Vector3dTwoTripleFunc EqualKey3 = bind(EqualKey<Vector3d, 3>, std::placeholders::_1, std::placeholders::_2);

    unordered_map<Vector3d, size_t, Vector3dTripleFunc, Vector3dTwoTripleFunc> map_vert(1, HashFunc3, EqualKey3);
    map<size_t, size_t>                                                        map_old_to_new_vert_id;

    string str_line;
    while (getline(f_in, str_line))
    {
        if (str_line.size() < 2)
            continue;

        if (str_line[0] == 'v'
            && str_line[1] == ' ')
        {
            Matrix<T, 3, 1> v;
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
                table_vert.push_back(v);
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
            Matrix<size_t, 3, 1> tri_vert;
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

            table_tri.push_back(tri_vert);
            continue;
        }
    }

    return 0;
}

template <typename T>
Eigen::Matrix<T, 3, 2> KdTree<T>::get_aabb(
    const std::vector<size_t>& table_tri) const
{
    Matrix<T, 3, 2> aabb;
    aabb << numeric_limits<T>::max(), -numeric_limits<T>::max(),
        numeric_limits<T>::max(), -numeric_limits<T>::max(),
        numeric_limits<T>::max(), -numeric_limits<T>::max();

    for (auto id_tri : table_tri)
    {
        const Matrix<T, 3, 3>& tri = get_tri(id_tri);
        for (size_t v = 0; v < 3; ++v)
        {
            for (size_t i = 0; i < 3; ++i)
            {
                if (tri(i, v) < aabb(i, 0))
                    aabb(i, 0) = tri(i, v);
                if (tri(i, v) > aabb(i, 1))
                    aabb(i, 1) = tri(i, v);
            }
        }
    }

    return aabb;
}

template <typename T>
int KdTree<T>::destory_tree(Node<T>* ptr)
{
    if (ptr != nullptr)
    {
        if (ptr->node_left == nullptr
            && ptr->node_right == nullptr)
        {
            delete ptr;
            ptr = nullptr;
            return 0;
        }
    }
    else
    {
        return 0;
    }
    assert(ptr->node_right != nullptr && ptr->node_left != nullptr);
    destory_tree(ptr->node_left);
    destory_tree(ptr->node_right);

    delete ptr;
    return 0;
}

template <typename T>
KdTree<T>::~KdTree()
{
    destory_tree(root);
    root = nullptr;
}

template <typename T>
size_t KdTree<T>::get_tri_num() const
{
    return table_tri.size();
}

template <typename T>
Eigen::Matrix<T, 3, 3> KdTree<T>::get_tri(size_t id_tri) const
{
    Matrix<size_t, 3, 1> tri_v = table_tri.at(id_tri);
    Matrix<T, 3, 3>      tri;
    for (size_t i = 0; i < 3; ++i)
    {
        tri.col(i) = table_vert.at(tri_v[i]);
    }

    return tri;
}

template <typename T>
std::vector<Eigen::Matrix<T, 3, 3>> KdTree<T>::get_neigh_tri(size_t id_tri) const
{
    Matrix<T, 3, 2> aabb = get_aabb(vector<size_t>(1, id_tri));

    vector<size_t> table_neigh_tri_id;
    get_neigh_tri(aabb, root, table_neigh_tri_id);

    auto is_find = find(table_neigh_tri_id.begin(), table_neigh_tri_id.end(), id_tri);
    assert(is_find != table_neigh_tri_id.end());
    table_neigh_tri_id.erase(is_find);
    vector<Matrix<T, 3, 3>> table_neigh_tri;
    transform(table_neigh_tri_id.begin(), table_neigh_tri_id.end(), back_inserter(table_neigh_tri), [this](size_t id_tri) {
        return get_tri(id_tri);
    });
    return table_neigh_tri;
}

template <typename T>
int KdTree<T>::get_neigh_tri(const Matrix<T, 3, 2>& aabb, const Node<T>* const ptr, std::vector<size_t>& table_neigh_tri) const
{
    const Matrix<T, 3, 2>& aabb_ptr = ptr->aabb;
    size_t                 axis     = 0;
    for (; axis < 3; ++axis)
    {
        if (aabb(axis, 0) <= aabb_ptr(axis, 1) && aabb(axis, 0) >= aabb_ptr(axis, 0))
            continue;
        if (aabb_ptr(axis, 0) <= aabb(axis, 1) && aabb_ptr(axis, 0) >= aabb(axis, 0))
            continue;

        break;
    }

    if (axis != 3)
        return 1;

    if (ptr->node_left != nullptr)
    {
        assert(ptr->node_right != nullptr);
        get_neigh_tri(aabb, ptr->node_left, table_neigh_tri);
        get_neigh_tri(aabb, ptr->node_right, table_neigh_tri);
    }
    else
    {
        assert(ptr->node_left == nullptr);
        assert(ptr->node_right == nullptr);
        assert(ptr->table_tri.size() == 1);
        size_t id_tri = ptr->table_tri.at(0);
        table_neigh_tri.push_back(id_tri);
    }

    return 0;
}

#endif