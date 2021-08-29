#pragma once

#include <yaml-cpp/yaml.h>

#include <core/math/vec.h>
#include <core/scene/scene2.h>

#include <iostream>
#include <string>

#include "yaml_utils.h"
#include "yaml_colliders.h"
#include "yaml_emitters.h"

namespace YAML {

template <>
struct convert<pbal::Scene2>
{
    static Node encode(const pbal::Scene2& rhs)
    {
        Node node;
        return node;
    }

    static bool decode(const Node& node, pbal::Scene2& scene)
    {
        if (!node.IsMap())
        {
            return false;
        }

        auto collidersNode = node["colliders"];
        for (auto node : collidersNode)
        {
            auto colliderNode = node["collider"];

            auto type = colliderNode["type"];
            if (!type.IsDefined())
                continue;

            auto name = type.as<std::string>();
            if (name == "polygon")
            {
                auto colliderPtr = std::make_shared<pbal::ColliderPolygon2>();
                *colliderPtr     = colliderNode.as<pbal::ColliderPolygon2>();
                scene.colliders.emplace_back(colliderPtr);
            }
            else if (name == "obj")
            {
            }
        }

        auto emittersNode = node["emitters"];
        for (auto node : emittersNode)
        {
            auto emitterNode = node["emitter"];

            auto type = emitterNode["type"];
            if (!type.IsDefined())
                continue;

            auto name = type.as<std::string>();
            if (name == "box")
            {
                auto emitterPtr = std::make_shared<pbal::ParticleEmitterBox2>();
                *emitterPtr     = emitterNode.as<pbal::ParticleEmitterBox2>();
                scene.emitters.emplace_back(emitterPtr);
            }
            else if (name == "sphere")
            {
                auto emitterPtr = std::make_shared<pbal::EmitterSphere2>();
                *emitterPtr     = emitterNode.as<pbal::EmitterSphere2>();
                scene.emitters.emplace_back(emitterPtr);
            }
        }

        return true;
    }
};

}  // namespace YAML