#pragma once
#include "Physika_Core/DataTypes.h"
#include "Physika_Framework/Framework/NumericalModel.h"
#include "Physika_Framework/Framework/FieldVar.h"
#include "Physika_Framework/Mapping/PointsToPoints.h"

namespace Physika
{
	class Mapping;

	template<typename TDataType>
	class ParticleSystem : public NumericalModel
	{
		DECLARE_CLASS_1(ParticleSystem, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleSystem();
		virtual ~ParticleSystem();

		bool execute() override;

		void updateTopology() override;

		void step(Real dt) override;

	protected:
		bool initializeImpl() override;

	private:
		std::shared_ptr<PointsToPoints<TDataType>> m_mapping;

		HostVariablePtr<size_t> m_num;
		HostVariablePtr<Real> m_mass;
		HostVariablePtr<Real> m_smoothingLength;
		HostVariablePtr<Real> m_samplingDistance;
		HostVariablePtr<Real> m_restDensity;

		HostVariablePtr<Coord> m_lowerBound;
		HostVariablePtr<Coord> m_upperBound;

		HostVariablePtr<Coord> m_gravity;
	};


#ifdef PRECISION_FLOAT
	template class ParticleSystem<DataType3f>;
#else
	template class ParticleSystem<DataType3d>;
#endif
}