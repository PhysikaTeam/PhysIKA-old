#pragma once
#include "Framework/Framework/ModuleConstraint.h"
#include "Framework/Framework/FieldArray.h"

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

		void clear();

		bool constrain() override;

		void setInitPositionID(FieldID id) { m_initPosID = id; }

	protected:
		FieldID m_initPosID;

	private:
		std::vector<int> m_ids;
		DeviceArray<int> m_device_ids;
	};

#ifdef PRECISION_FLOAT
template class FixedPoints<DataType3f>;
#else
template class FixedPoints<DataType3d>;
#endif

}
