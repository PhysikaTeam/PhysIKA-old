#pragma once
#include "Core/Array/Array.h"
#include "Framework/Framework/ModuleConstraint.h"
#include "Framework/Framework/FieldVar.h"
#include "Framework/Framework/FieldArray.h"
#include "Framework/Topology/FieldNeighbor.h"
#include "Kernel.h"
#include "Framework/Framework/ModuleTopology.h"

namespace PhysIKA {

	template<typename TDataType> class DensitySummationMesh;

	/*!
	*	\class	DensityPBDMesh
	*	\brief	This class implements a position-based solver for incompressibility.
	*/
	template<typename TDataType>
	class DensityPBDMesh : public ConstraintModule
	{
		DECLARE_CLASS_1(DensityPBDMesh, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;

		DensityPBDMesh();
		~DensityPBDMesh() override;

		bool constrain() override;

		void takeOneIteration();

		void updateVelocity();

		void setIterationNumber(int n) { m_maxIteration = n; }

		DeviceArray<Real>& getDensity() { return m_density.getValue(); }

	protected:
		bool initializeImpl() override;

	public:
		VarField<Real> m_restDensity;
		VarField<Real> m_smoothingLength;

		DeviceArrayField<Coord> m_position;
		DeviceArrayField<Coord> m_velocity;
		DeviceArrayField<Real> m_massInv; // mass^-1 as described in unified particle physics
		DeviceArrayField<Real> m_veln;

		NeighborField<int> m_neighborhood;

		NeighborField<int> m_neighborhoodTri;
		DeviceArrayField<Coord> TriPoint;
		DeviceArrayField<Triangle> Tri;

		DeviceArrayField<Real> m_density;

		VarField<Real> sampling_distance;
		VarField<int> use_mesh;
		VarField<int> use_ghost;

		VarField<int> Start;


	private:
		int m_maxIteration;

		SpikyKernel<Real> m_kernel;

		DeviceArray<Real> m_lamda;
		DeviceArray<Coord> m_deltaPos;
		DeviceArray<Coord> m_position_old;

		std::shared_ptr<DensitySummationMesh<TDataType>> m_densitySum;
	};



}