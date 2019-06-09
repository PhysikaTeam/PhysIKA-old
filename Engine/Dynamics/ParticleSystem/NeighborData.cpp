#include "NeighborData.h"
#include "Core/DataTypes.h"
#include "Framework/Topology/FieldNeighbor.h"
namespace Physika
{
	template class NeighborField<int>;
	template class NeighborField<TPair<DataType3f>>;
	template class NeighborField<TPair<DataType3d>>;
}