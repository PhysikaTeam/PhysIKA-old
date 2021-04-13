#pragma once
#include "Framework/Framework/ModuleCompute.h"
#include "Framework/Framework/FieldVar.h"
#include "Framework/Framework/FieldArray.h"
#include "Framework/Topology/FieldNeighbor.h"

namespace PhysIKA {

	/**
	 * @brief The standard summation density
	 * 
	 * @tparam TDataType 
	 */
	template<typename TDataType>
	class SummationDensity : public ComputeModule
	{
		DECLARE_CLASS_1(SummationDensity, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		SummationDensity();
		~SummationDensity() override {};

		void compute() override;
	
	protected:
		void calculateScalingFactor();
		void calculateParticleMass();

		void compute(DeviceArray<Real>& rho);

		void compute(
			DeviceArray<Real>& rho,
			DeviceArray<Coord>& pos,
			NeighborList<int>& neighbors,
			Real smoothingLength,
			Real mass);

	public:
		DEF_EMPTY_VAR(RestDensity, Real, "Rest Density");
		DEF_EMPTY_VAR(SmoothingLength, Real, "Indicating the smoothing length");
		DEF_EMPTY_VAR(SamplingDistance, Real, "Indicating the initial sampling distance");

		///Define inputs
		/**
		 * @brief Particle positions
		 */
		DEF_EMPTY_IN_ARRAY(Position, Coord, DeviceType::GPU, "Particle position");

		/**
		 * @brief Neighboring particles
		 *
		 */
		DEF_EMPTY_IN_NEIGHBOR_LIST(NeighborIndex, int, "Neighboring particles' ids");

		///Define outputs
		/**
		 * @brief Particle densities
		 */
		DEF_EMPTY_OUT_ARRAY(Density, Real, DeviceType::GPU, "Particle position");

	private:
		Real m_particle_mass;
		Real m_factor;
	};

#ifdef PRECISION_FLOAT
	template class SummationDensity<DataType3f>;
#else
	template class SummationDensity<DataType3d>;
#endif
}