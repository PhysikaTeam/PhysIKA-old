/**
 * this is a bug between nvcc 10.1 and gcc7.2
 * the intermediate file between nvcc and gcc has error, this will cause the function "getReference()" getting into the dead loop.
 */

#include "FieldNeighbor.h"
#include <Core/DataTypes.h>

namespace Physika{

template<typename T>
std::shared_ptr<NeighborList<T>> NeighborField<T>::getReference()
{
	Field* source = getSource();
	if (source == nullptr)
	{
		return m_data;
	}
	else
	{
		NeighborField<T>* var = dynamic_cast<NeighborField<T>*>(source);
		if (var != nullptr)
		{
			return var->getReference();
			//return (*var).getReference();
		}
		else
		{
			return nullptr;
		}
	}
}

template class NeighborField<int>;
template class NeighborField<TPair<DataType3f>>;
}