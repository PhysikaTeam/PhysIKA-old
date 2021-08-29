#pragma once

#include <yaml-cpp/yaml.h>

#include <core/math/vec.h>

#include <string>

namespace YAML {

// pbal::Vec2d
template <>
struct convert<pbal::Vec2d>
{
    static Node encode(const pbal::Vec2d& rhs)
    {
        Node node;
        node.push_back(rhs.x);
        node.push_back(rhs.y);
        return node;
    }

    static bool decode(const Node& node, pbal::Vec2d& rhs)
    {
        if (!node.IsSequence() || node.size() != 2)
        {
            return false;
        }

        rhs.x = node[0].as<double>();
        rhs.y = node[1].as<double>();
        return true;
    }
};

// pbal::Vec3d
template <>
struct convert<pbal::Vec3d>
{
    static Node encode(const pbal::Vec3d& rhs)
    {
        Node node;
        node.push_back(rhs.x);
        node.push_back(rhs.y);
        node.push_back(rhs.z);
        return node;
    }

    static bool decode(const Node& node, pbal::Vec3d& rhs)
    {
        if (!node.IsSequence() || node.size() != 3)
        {
            return false;
        }

        rhs.x = node[0].as<double>();
        rhs.y = node[1].as<double>();
        rhs.z = node[2].as<double>();
        return true;
    }
};

}  // namespace YAML