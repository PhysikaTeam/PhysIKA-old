#pragma once
#include "Core/Platform.h"
#include <typeinfo>
#include <string>
#include <cuda_runtime.h>
#include "FieldVar.h"
#include "FieldArray.h"
#include "Module.h"
#include "../Topology/FieldNeighbor.h"


namespace Physika {
/*!
*	\class	DataFlow
*	\brief	Control the data stream.
*/
class DataFlow
{
public:

// 	template<typename T>
// 	static bool connect(VarField<T>& var, Slot<T>& slot)
// 	{
// 		slot.setReference(var.getReference());
// 		return true;
// 	}
// 
// 	template<typename T, DeviceType deviceType>
// 	static bool connect(ArrayField<T, deviceType>& var, Slot<Array<T, deviceType>>& slot)
// 	{
// 		slot.setReference(var.getReference());
// 		return true;
// 	}
// 
// 	template<typename T>
// 	static bool connect(NeighborField<T>& var, Slot<NeighborList<T>>& slot)
// 	{
// 		slot.setReference(var.getReference());
// 		return true;
// 	}

private:
};

}