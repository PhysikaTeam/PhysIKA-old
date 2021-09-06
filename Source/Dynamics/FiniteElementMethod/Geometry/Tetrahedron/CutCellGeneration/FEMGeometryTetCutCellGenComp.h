#pragma once

#include <Eigen/Core>

/**
 * @brief Compare two 3d vertexs
 * 
 */
struct VertComp
{
    bool operator()(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2) const
    {
        if (v1(0) < v2(0))
            return true;
        else if (v1(0) == v2(0) && v1(1) < v2(1))
            return true;
        else if (v1(0) == v2(0) && v1(1) == v2(1) && v1(2) < v2(2))
            return true;
        else
            return false;
    }
};

/**
 * @brief Compare two 2d vertexs
 * 
 */
struct Vert2DComp
{
    bool operator()(const Eigen::Vector2d& v1, const Eigen::Vector2d& v2) const
    {
        if (v1(0) < v2(0))
            return true;
        else if (v1(0) == v2(0) && v1(1) < v2(1))
            return true;
        else
            return false;
    }
};

typedef Eigen::Matrix<size_t, 3, 1> Vector3st;

/**
 * @brief Compare two ids
 * 
 */
struct IdComp
{
    bool operator()(const Vector3st& id1, const Vector3st& id2) const
    {
        if (id1(0) < id2(0))
            return true;
        else if (id1(0) == id2(0) && id1(1) < id2(1))
            return true;
        else if (id1(0) == id2(0) && id1(1) == id2(1) && id1(2) < id2(2))
            return true;
        else
            return false;
    }
};