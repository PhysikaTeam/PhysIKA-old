#include "NeighborData.h"
#include "Core/DataTypes.h"
#include "Framework/Topology/FieldNeighbor.h"
namespace PhysIKA {
template class NeighborField<int>;
template class NeighborField<TPair<DataType3f>>;
template class NeighborField<TPair<DataType3d>>;
}  // namespace PhysIKA