#include "CollisionModel.h"
#include "Framework/Framework/Node.h"

namespace PhysIKA {

CollisionModel::CollisionModel()
{
}

CollisionModel::~CollisionModel()
{
}

bool CollisionModel::execute()
{
    this->doCollision();

    return true;
}

}  // namespace PhysIKA