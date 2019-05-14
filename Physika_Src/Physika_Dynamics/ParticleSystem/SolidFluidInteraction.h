#pragma once
#include "Physika_Framework/Framework/Node.h"

namespace Physika
{
	template <typename T> class RigidBody;
	template <typename T> class ParticleSystem;

	/*!
	*	\class	SolidFluidInteraction
	*	\brief	Position-based fluids.
	*
	*	This class implements a position-based fluid solver.
	*	Refer to Macklin and Muller's "Position Based Fluids" for details
	*
	*/

	template<typename TDataType>
	class SolidFluidInteraction : public Node
	{
		DECLARE_CLASS_1(SolidFluidInteraction, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		SolidFluidInteraction();
		SolidFluidInteraction(std::string name);
		~SolidFluidInteraction() override;

	public:
		bool initialize() override;

		bool addRigidBody(std::shared_ptr<RigidBody<TDataType>> child);
		bool addParticleSystem(std::shared_ptr<ParticleSystem<TDataType>> child);

	private:
		void construct();

		std::list<std::shared_ptr<RigidBody<TDataType>>> m_rigids;
		std::list<std::shared_ptr<ParticleSystem<TDataType>>> m_particleSystems;
	};


#ifdef PRECISION_FLOAT
	template class SolidFluidInteraction<DataType3f>;
#else
	template class SolidFluidInteraction<DataType3d>;
#endif
}