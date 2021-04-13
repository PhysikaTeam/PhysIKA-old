#pragma once
#include "Framework/Framework/ModuleCompute.h"
#include "Framework/Framework/FieldVar.h"
#include "Framework/Framework/FieldArray.h"
#include "Framework/Topology/FieldNeighbor.h"
#include "Framework/Framework/ModuleTopology.h"

namespace PhysIKA {

	template<typename TDataType> class NeighborList;
	template <typename TDataType> class TriangleSet;

	template<typename TDataType>
	class DensitySummationMesh : public ComputeModule
	{
		DECLARE_CLASS_1(DensitySummation, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;

		DensitySummationMesh();
		~DensitySummationMesh() override {};

		void compute() override;

		void compute(DeviceArray<Real>& rho);

		void compute(
			DeviceArray<Real>& rho,
			DeviceArray<Coord>& pos,
			DeviceArray<TopologyModule::Triangle>& Tri,
			DeviceArray<Coord>& positionTri,
			NeighborList<int>& neighbors,
			NeighborList<int>& neighborsTri,
			Real smoothingLength,
			Real mass,
			Real sampling_distance,
			int use_mesh,
			int use_ghost,
			int Start);

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
		NeighborField<int> m_neighborhoodTri;
		DeviceArrayField<Coord> TriPoint;
		DeviceArrayField<Triangle> Tri;

		VarField<Real> sampling_distance;
		VarField<int> use_mesh;
		VarField<int> use_ghost;
		VarField<int> Start;

	private:
		Real m_factor;
	};

#ifdef PRECISION_FLOAT
	template class DensitySummationMesh<DataType3f>;
#else
	template class DensitySummationMesh<DataType3d>;
#endif
}