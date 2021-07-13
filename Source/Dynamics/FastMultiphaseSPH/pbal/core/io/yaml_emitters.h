#pragma once

#include <yaml-cpp/yaml.h>

#include <core/math/vec.h>
#include <core/scene/emitters/emitter_sphere2.h>
#include <core/scene/emitters/particle_emitter_box2.h>

#include <iostream>
#include <string>

#include "yaml_utils.h"

namespace YAML {

template <>
struct convert<pbal::ParticleEmitterBox2>
{
    static Node encode(const pbal::ParticleEmitterBox2& rhs)
    {
        Node node;
        return node;
    }

    static bool decode(const Node& node, pbal::ParticleEmitterBox2& emitter)
    {
        if (!node.IsMap() || node.size() < 3)
        {
            return false;
        }
        emitter.type = "box";

        emitter.center   = node["center"].as<pbal::Vec2d>();
        emitter.distance = node["distance"].as<pbal::Vec2d>();
        emitter.r        = node["r"].as<double>();

        return true;
    }
};

template <>
struct convert<pbal::EmitterSphere2>
{
    static Node encode(const pbal::EmitterSphere2& rhs)
    {
        Node node;
        return node;
    }

    static bool decode(const Node& node, pbal::EmitterSphere2& emitter)
    {
        if (!node.IsMap() || node.size() < 3)
        {
            return false;
        }
        emitter.type = "sphere";

        emitter.center = node["center"].as<pbal::Vec2d>();
        emitter.radius = node["radius"].as<double>();
        emitter.r      = node["r"].as<double>();

        return true;
    }
};

}  // namespace YAML