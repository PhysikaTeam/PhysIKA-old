#pragma once
#include <vector_types.h>
#include <vector>
#include "Physika_Framework/Framework/NumericalModel.h"
#include "ElasticityModule.h"
#include "Physika_Framework/Framework/FieldVar.h"


namespace Physika
{
	template<typename> class PointsToPoints;
	template<typename> class ParticleIntegrator;
	template<typename> class ElasticityModule;

	/*!
	*	\class	ParticleSystem
	*	\brief	Projective peridynamics
	*
	*	This class implements the projective peridynamics.
	*	Refer to He et al' "Projective peridynamics for modeling versatile elastoplastic materials" for details.
	*/
	template<typename TDataType>
	class Peridynamics : public NumericalModel
	{
		DECLARE_CLASS_1(Peridynamics, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		Peridynamics();
		~Peridynamics() override {};

		/*!
		*	\brief	All variables should be set appropriately before initializeImpl() is called.
		*/
		bool initializeImpl() override;

		void updateTopology() override;

		void step(Real dt) override;

	private:
		std::shared_ptr<PointsToPoints<TDataType>> m_mapping;
		std::shared_ptr<ParticleIntegrator<TDataType>> prediction;
		std::shared_ptr<ElasticityModule<TDataType>>  m_elasticity;

		HostVariablePtr<int> m_num;
		HostVariablePtr<Real> m_mass;
		HostVariablePtr<Real> m_smoothingLength;
		HostVariablePtr<Real> m_samplingDistance;
		HostVariablePtr<Real> m_restDensity;
	};

#ifdef PRECISION_FLOAT
	template class Peridynamics<DataType3f>;
#else
	template class ParticleFluid<DataType3d>;
#endif
}