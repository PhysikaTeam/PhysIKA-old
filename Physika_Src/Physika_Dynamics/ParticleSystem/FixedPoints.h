#pragma once
#include "Physika_Core/Platform.h"
#include "Physika_Framework/Framework/ModuleConstraint.h"
#include "Physika_Core/DataTypes.h"
#include "Physika_Framework/Framework/FieldArray.h"

namespace Physika {

	template<typename TDataType>
	class FixedPoints : public ConstraintModule
	{
		DECLARE_CLASS_1(FixedPoints, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		FixedPoints();
		~FixedPoints() override;
		
		void addPoint(int id);

		void constrain() override;

	protected:
		std::vector<int> m_ids;
		DeviceArray<int> m_device_ids;
	};

#ifdef PRECISION_FLOAT
template class FixedPoints<DataType3f>;
#else
template class FixedPoints<DataType3d>;
#endif

}
