#pragma once
#include "Framework/Framework/ModuleCompute.h"
#include "Framework/Framework/FieldVar.h"
#include "Framework/Framework/FieldArray.h"
#include "Framework/Topology/FieldNeighbor.h"

namespace PhysIKA {

	template<typename TDataType> class NeighborList;

	template<typename TDataType>
	class DensitySummation : public ComputeModule
	{
		DECLARE_CLASS_1(DensitySummation, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		DensitySummation();
		~DensitySummation() override {};

		void compute() override;

		void compute(DeviceArray<Real>& rho);

		void compute(
			DeviceArray<Real>& rho,
			DeviceArray<Coord>& pos,
			NeighborList<int>& neighbors,
			Real smoothingLength,
			Real mass);

		void setCorrection(Real factor) { m_factor = factor; }
	
	protected:
		bool initializeImpl() override;

		void calculateScalingFactor();

	public:
		VarField<Real> m_mass;
		VarField<Real> m_restDensity;
		
		DEF_EMPTY_PARAM_VAR(SmoothingLength, Real, "Indicating the smoothing length");
		DEF_EMPTY_PARAM_VAR(SamplingDistance, Real, "Indicating the initial sampling distance");

		DeviceArrayField<Coord> m_position;
		DeviceArrayField<Real> m_density;

		NeighborField<int> m_neighborhood;

	private:
		Real m_factor;
	};

#ifdef PRECISION_FLOAT
	template class DensitySummation<DataType3f>;
#else
	template class DensitySummation<DataType3d>;
#endif
}