/**
 * this is a bug between nvcc 10.1 and gcc7.2
 * the intermediate file between nvcc and gcc has error, this will cause the function "getReference()" getting into the dead loop.
 */

#include "FieldNeighbor.h"
#include <Core/DataTypes.h>

namespace PhysIKA{

}