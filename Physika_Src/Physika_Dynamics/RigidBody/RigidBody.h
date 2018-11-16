#pragma once
#include <vector_types.h>
#include <vector>
#include "Physika_Core/DataTypes.h"
#include "Physika_Framework/Framework/FieldVar.h"
#include "Physika_Framework/Framework/NumericalModel.h"
#include "Physika_Core/Quaternion/quaternion.h"
#include "Physika_Framework/Mapping/RigidToPoints.h"

namespace Physika
{
	template<typename TDataType>
	class RigidBody : public NumericalModel
	{
		DECLARE_CLASS_1(Peridynamics, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef typename TDataType::Rigid Rigid;
		typedef typename TDataType::Rigid::RotationDOF RotateCoord;

		RigidBody();
		~RigidBody() override {};

		/*!
		*	\brief	All variables should be set appropriately before Initliazed() is called.
		*/
		bool initializeImpl() override;

		void updateTopology() override;

		void step(Real dt) override;

	private:
		std::shared_ptr<RigidToPoints<TDataType>> m_mapping;

		Quaternion<Real> m_quaternion;

		Coord m_displacement;
		Matrix m_deltaRotation;
	};

#ifdef PRECISION_FLOAT
	template class RigidBody<DataType3f>;
#else
	template class RigidBody<DataType3d>;
#endif
}