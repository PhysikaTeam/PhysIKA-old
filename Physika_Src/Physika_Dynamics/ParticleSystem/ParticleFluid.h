#pragma once
#include "Physika_Framework/Framework/NumericalModel.h"
#include "Physika_Framework/Framework/FieldVar.h"

namespace Physika
{
	template<typename TDataType> class PointsToPoints;
	template<typename TDataType> class ParticleIntegrator;
	template<typename TDataType> class NeighborQuery;
	/*!
	*	\class	ParticleSystem
	*	\brief	Position-based fluids.
	*
	*	This class implements a position-based fluid solver.
	*	Refer to Macklin and Muller's "Position Based Fluids" for details
	*
	*/
	template<typename TDataType>
	class ParticleFluid : public NumericalModel
	{
		DECLARE_CLASS_1(ParticleFluid, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleFluid();
		virtual ~ParticleFluid();

		void updateTopology() override;

		void step(Real dt) override;

		void setSmoothingLength(Real len) { m_smoothingL = len; }
		void setSamplingDistance(Real distance) { m_samplingD = distance; }
		void setRestDensity(Real rho) { m_restRho = rho; }

	protected:
		bool initializeImpl() override;

	private:
		std::shared_ptr<PointsToPoints<TDataType>> m_mapping;
		std::shared_ptr<ParticleIntegrator<TDataType>> m_integrator;
		std::shared_ptr<NeighborQuery<TDataType>>m_nbrQuery;

		int m_pNum;
		Real m_smoothingL;
		Real m_samplingD;
		Real m_restRho;

		HostVariablePtr<Real> m_smoothingLength;
		HostVariablePtr<Real> m_samplingDistance;
		HostVariablePtr<Real> m_restDensity;

		HostVariablePtr<Coord> m_lowerBound;
		HostVariablePtr<Coord> m_upperBound;
	};


#ifdef PRECISION_FLOAT
	template class ParticleFluid<DataType3f>;
#else
	template class ParticleFluid<DataType3d>;
#endif
}