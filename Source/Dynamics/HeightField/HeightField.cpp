#include "HeightField.h"
#include "Framework/Topology/Frame.h"
#include "Framework/Topology/PointSet.h"
#include "Framework/Topology/TriangleSet.h"
#include "Framework/Mapping/FrameToPointSet.h"
#include "Rendering/SurfaceMeshRender.h"

namespace PhysIKA
{
	IMPLEMENT_CLASS_1(HeightField, TDataType)

	template<typename TDataType>
	HeightField<TDataType>::HeightField()
		: Node()
	{
	}

	template<typename TDataType>
	HeightField<TDataType>::~HeightField()
	{
		
	}

	template<typename TDataType>
	bool HeightField<TDataType>::initialize()
	{
		return true;
	}
}