#pragma once
#include "Physika_Framework/Framework/ModuleCompute.h"
#include "Physika_Framework/Framework/FieldVar.h"
#include "Physika_Framework/Framework/FieldArray.h"
#include "Physika_Framework/Topology/FieldNeighbor.h"

namespace Physika {

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
		void setSmoothingLength(Real length) { m_smoothingLength.setValue(length); }
	
	protected:
		bool initializeImpl() override;

	public:
		VarField<Real> m_mass;
		VarField<Real> m_restDensity;
		VarField<Real> m_smoothingLength;

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