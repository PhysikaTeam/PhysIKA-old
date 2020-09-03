#pragma once
#include "Framework/Framework/Node.h"
#include "Framework/Framework/FieldArray.h"

#include "Dynamics/RigidBody/RigidBody.h"
#include "Dynamics/ParticleSystem/ParticleSystem.h"

namespace PhysIKA {

	template <typename T> class TriangleSet;
	template <typename T> class NeighborQuery;

	template<typename TDataType>
	class MeshBoundary : public Node
	{
		DECLARE_CLASS_1(MeshBoundary, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;

		MeshBoundary();
		~MeshBoundary() override;


		void loadMesh(std::string filename);

		void advance(Real dt) override;

		bool initialize() override;
		bool resetStatus() override;

	public:

		std::vector<std::shared_ptr<TriangleSet<TDataType>>> m_obstacles;

		std::vector<std::shared_ptr<RigidBody<TDataType>>> m_rigids;
		std::vector<std::shared_ptr<ParticleSystem<TDataType>>> m_particleSystems;

		DEF_NODE_PORTS(RigidBody, RigidBody<TDataType>, "A rigid body");
		DEF_NODE_PORTS(ParticleSystem, ParticleSystem<TDataType>, "Particle Systems");

	private:
		std::shared_ptr<NeighborQuery<TDataType>> m_nbrQuery;
		DeviceArrayField<Coord> point_positions;
		DeviceArrayField<Coord> point_velocities;
		DeviceArrayField<Coord> triangle_positions;
		DeviceArrayField<Triangle> triangle_index;
		VarField<Real> radius;

	};


#ifdef PRECISION_FLOAT
template class MeshBoundary<DataType3f>;
#else
template class MeshBoundary<DataType3d>;
#endif

}
