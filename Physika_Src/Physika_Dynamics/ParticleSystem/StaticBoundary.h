#pragma once
#include "Physika_Framework/Framework/Node.h"
#include "Physika_Framework/Framework/FieldArray.h"

namespace Physika {
	template <typename TDataType> class RigidBody;
	template <typename TDataType> class ParticleSystem;
	template <typename TDataType> class DistanceField3D;

	template<typename TDataType>
	class StaticBoundary : public Node
	{
		DECLARE_CLASS_1(StaticBoundary, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		StaticBoundary();
		~StaticBoundary() override;

		bool addRigidBody(std::shared_ptr<RigidBody<TDataType>> child);
		bool addParticleSystem(std::shared_ptr<ParticleSystem<TDataType>> child);

		void advance(Real dt) override;

	public:

		std::vector<std::shared_ptr<RigidBody<TDataType>>> m_rigids;
		std::vector<std::shared_ptr<ParticleSystem<TDataType>>> m_particleSystems;
	};

#ifdef PRECISION_FLOAT
template class StaticBoundary<DataType3f>;
#else
template class StaticBoundary<DataType3d>;
#endif

}
