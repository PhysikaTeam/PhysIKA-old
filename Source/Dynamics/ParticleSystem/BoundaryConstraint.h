#pragma once
#include "Framework/Framework/ModuleConstraint.h"
#include "Framework/Framework/FieldArray.h"

namespace PhysIKA {

	template<typename TDataType> class DistanceField3D;

	template<typename TDataType>
	class BoundaryConstraint : public ConstraintModule
	{
		DECLARE_CLASS_1(BoundaryConstraint, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		BoundaryConstraint();
		~BoundaryConstraint() override;

		bool constrain() override;

		bool constrain(DeviceArray<Coord>& position, DeviceArray<Coord>& velocity, Real dt);

		void load(std::string filename, bool inverted = false);
		void setCube(Coord lo, Coord hi, Real distance, bool inverted = false);
		void setSphere(Coord center, Real r, Real distance, bool inverted = false);

	public:
		DeviceArrayField<Coord> m_position;
		DeviceArrayField<Coord> m_velocity;

		Real m_normal_friction = 0.95f;
		Real m_tangent_friction = 0.0;

		std::shared_ptr<DistanceField3D<TDataType>> m_cSDF;
	};

#ifdef PRECISION_FLOAT
template class BoundaryConstraint<DataType3f>;
#else
template class BoundaryConstraint<DataType3d>;
#endif

}
