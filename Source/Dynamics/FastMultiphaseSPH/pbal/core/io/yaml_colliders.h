#pragma once

#include <yaml-cpp/yaml.h>

#include <core/math/vec.h>
#include <core/scene/colliders/collider_polygon2.h>

#include <iostream>
#include <string>

#include "yaml_utils.h"

namespace YAML {

template <>
struct convert<pbal::ColliderPolygon2>
{
    static Node encode(const pbal::ColliderPolygon2& rhs)
    {
        Node node;
        return node;
    }

    static bool decode(const Node& node, pbal::ColliderPolygon2& collider)
    {
        if (!node.IsMap() || node.size() < 2)
        {
            return false;
        }
        collider.type = "polygon";

        auto vertices = node["vertices"];
        for (auto v : vertices)
        {
            collider.polygon.push_back(v.as<pbal::Vec2d>());
        }
        if (node["w"].IsDefined())
        {
            collider.w = node["w"].as<double>();
        }
        if (node["vel"].IsDefined())
        {
            collider.vel = node["vel"].as<pbal::Vec2d>();
        }
        if (node["center"].IsDefined())
        {
            collider.center = node["center"].as<pbal::Vec2d>();
        }
        return true;
    }
};

}  // namespace YAML