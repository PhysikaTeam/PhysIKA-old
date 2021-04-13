#pragma once
#include "Framework/Framework/Node.h"
#include "Framework/Framework/FieldArray.h"

#include "Dynamics/RigidBody/RigidBody.h"
#include "Dynamics/ParticleSystem/ParticleSystem.h"

namespace PhysIKA {
// 	template <typename TDataType> class RigidBody;
// 	template <typename TDataType> class ParticleSystem;
	template <typename TDataType> class DistanceField3D;
	template <typename TDataType> class BoundaryConstraint;

	template<typename TDataType>
	class StaticBoundary : public Node
	{
		DECLARE_CLASS_1(StaticBoundary, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		StaticBoundary();
		~StaticBoundary() override;

//		bool addRigidBody(std::shared_ptr<RigidBody<TDataType>> child);
//		bool addParticleSystem(std::shared_ptr<ParticleSystem<TDataType>> child);

		void advance(Real dt) override;

		void loadSDF(std::string filename, bool bOutBoundary = false);
		void loadCube(Coord lo, Coord hi, Real distance = 0.005f, bool bOutBoundary = false, bool bVisible = false);
		void loadShpere(Coord center, Real r, Real distance = 0.005f, bool bOutBoundary = false, bool bVisible = false);

		void translate(Coord t);
		void scale(Real s);

	public:

		std::vector<std::shared_ptr<BoundaryConstraint<TDataType>>> m_obstacles;

		std::vector<std::shared_ptr<RigidBody<TDataType>>> m_rigids;
		std::vector<std::shared_ptr<ParticleSystem<TDataType>>> m_particleSystems;

		DEF_NODE_PORTS(RigidBody, RigidBody<TDataType>, "A rigid body");
		DEF_NODE_PORTS(ParticleSystem, ParticleSystem<TDataType>, "Particle Systems");
	};


#ifdef PRECISION_FLOAT
template class StaticBoundary<DataType3f>;
#else
template class StaticBoundary<DataType3d>;
#endif

}
