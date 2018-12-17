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

		void setMassID(FieldID id) { m_massID = id; }
		void setDensityID(FieldID id) { m_rhoID = id; }
		void setPositionID(FieldID id) { m_posID = id; }
		void setNeighborhoodID(FieldID id) { m_neighborID = id; }

		void setCorrection(Real factor) { m_factor = factor; }
		void setSmoothingLength(Real length) { m_smoothingLength = length; }
	
	protected:
		FieldID m_massID;
		FieldID m_rhoID;
		FieldID m_posID;
		FieldID m_neighborID;

	private:
		Real m_factor;
		Real m_smoothingLength;
	};

#ifdef PRECISION_FLOAT
	template class DensitySummation<DataType3f>;
#else
	template class DensitySummation<DataType3d>;
#endif
}