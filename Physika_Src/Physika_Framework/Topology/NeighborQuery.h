#pragma once
#include "Physika_Framework/Framework/ModuleCompute.h"
#include "Physika_Framework/Framework/FieldVar.h"
#include "Physika_Framework/Framework/FieldArray.h"
#include "Physika_Framework/Topology/GridHash.h"

namespace Physika {
	template<typename ElementType> class NeighborList;

	template<typename TDataType>
	class NeighborQuery : public ComputeModule
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		NeighborQuery();
		NeighborQuery(Real s, Coord lo, Coord hi);
		~NeighborQuery() override {};
		
		void compute() override;

		void queryParticleNeighbors(NeighborList<int>& nbr,  DeviceArray<Coord>& pos, Real radius);

		void setRadius(Real r) { m_radius = r; }

	private:
		void queryNeighborSize(DeviceArray<int>& num, DeviceArray<Coord>& pos, Real h);
		void queryNeighborDynamic(NeighborList<int>& nbrList, DeviceArray<Coord>& pos, Real h);

		void queryNeighborFixed(NeighborList<int>& nbrList, DeviceArray<Coord>& pos, Real h);

	protected:
		FieldID m_posID;
		FieldID m_adaptNID;

	private:
		int m_maxNum;
		Real m_radius;

		GridHash<TDataType> hash;
	};

#ifdef PRECISION_FLOAT
	template class NeighborQuery<DataType3f>;
#else
	template class NeighborQuery<DataType3d>;
#endif
}