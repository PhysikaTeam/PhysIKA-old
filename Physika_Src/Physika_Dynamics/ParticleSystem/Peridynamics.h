#pragma once
#include <vector_types.h>
#include <vector>
#include "Physika_Core/DataTypes.h"
#include "Physika_Framework/Framework/NumericalModel.h"
#include "Physika_Framework/Topology/INeighbors.h"
#include "ParticlePrediction.h"
#include "ElasticityModule.h"
#include "Framework/FieldVar.h"
#include "Mapping/PointsToPoints.h"

namespace Physika
{
	template<typename TDataType>
	class Peridynamics : public NumericalModel
	{
		DECLARE_CLASS_1(Peridynamics, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TRestShape<TDataType> RestShape;

		Peridynamics();
		~Peridynamics() override {};

		/*!
		*	\brief	All variables should be set appropriately before Initliazed() is called.
		*/
		bool initializeImpl() override;

		void updateTopology() override;

		void step(Real dt) override;

	private:
		std::shared_ptr<PointsToPoints<TDataType>> m_mapping;

		std::shared_ptr<ParticlePrediction<TDataType>> prediction;
		std::shared_ptr<ElasticityModule<TDataType>> elasticity;

		HostVariablePtr<size_t> m_num;
		HostVariablePtr<Real> m_mass;
		HostVariablePtr<Real> m_smoothingLength;
		HostVariablePtr<Real> m_samplingDistance;
		HostVariablePtr<Real> m_restDensity;
	};

#ifdef PRECISION_FLOAT
	template class Peridynamics<DataType3f>;
#else
	template class ParticleSystem<DataType3d>;
#endif
}