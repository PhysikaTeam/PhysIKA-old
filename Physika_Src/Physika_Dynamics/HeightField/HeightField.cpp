#include "HeightField.h"
#include "Physika_Framework/Topology/Frame.h"
#include "Physika_Framework/Topology/PointSet.h"
#include "Physika_Framework/Topology/TriangleSet.h"
#include "Physika_Framework/Mapping/FrameToPointSet.h"
#include "Physika_Render/SurfaceMeshRender.h"

namespace Physika
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