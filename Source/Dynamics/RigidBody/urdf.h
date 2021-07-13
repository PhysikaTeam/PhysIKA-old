#pragma once

#include <string>
#include "Dynamics/RigidBody/RigidBodyRoot.h"

//#include <vector>
#include <Core/Vector/vector_3d.h>
//#include <Framework/Framework/ModuleTopology.h>

namespace PhysIKA {

class Urdf
{
public:
    RigidBodyRoot_ptr loadFile(std::string filename);
};

}  // namespace PhysIKA