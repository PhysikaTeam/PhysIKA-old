#pragma once
#include "Framework/Framework/ModuleCompute.h"
#include "Framework/Framework/FieldVar.h"
#include "Framework/Framework/FieldArray.h"
#include "Framework/Topology/FieldNeighbor.h"
#include "Framework/Topology/GridHash.h"
#include "Core/Utility.h"

namespace Physika {
	template<typename ElementType> class NeighborList;

	template<typename TDataType>
	class NeighborQuery : public ComputeModule
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		NeighborQuery();
		NeighborQuery(DeviceArray<Coord>& position);
		NeighborQuery(Real s, Coord lo, Coord hi);
		~NeighborQuery() override;
		
		void compute() override;

		void setRadius(Real r) { m_radius.setValue(r); }
		void setBoundingBox(Coord lowerBound, Coord upperBound);

		void queryParticleNeighbors(NeighborList<int>& nbr, DeviceArray<Coord>& pos, Real radius);

		void setNeighborSizeLimit(int num) { m_maxNum = num; }

		NeighborList<int>& getNeighborList() { return m_neighborhood.getValue(); }

	protected:
		bool initializeImpl() override;

	private:
		void queryNeighborSize(DeviceArray<int>& num, DeviceArray<Coord>& pos, Real h);
		void queryNeighborDynamic(NeighborList<int>& nbrList, DeviceArray<Coord>& pos, Real h);

		void queryNeighborFixed(NeighborList<int>& nbrList, DeviceArray<Coord>& pos, Real h);

	public:
		VarField<Real> m_radius;

		DeviceArrayField<Coord> m_position;
		NeighborField<int> m_neighborhood;

	private:
		int m_maxNum;

		Coord m_lowBound;
		Coord m_highBound;

		GridHash<TDataType> m_hash;

		int* m_ids;
		Real* m_distance;

		Reduction<int>* m_reduce;
		Scan m_scan;
	};

#ifdef PRECISION_FLOAT
	template class NeighborQuery<DataType3f>;
#else
	template class NeighborQuery<DataType3d>;
#endif
}