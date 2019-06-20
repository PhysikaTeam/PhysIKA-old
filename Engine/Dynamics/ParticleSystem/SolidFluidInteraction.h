#pragma once
#include "Framework/Framework/Node.h"

namespace Physika
{
	template <typename T> class RigidBody;
	template <typename T> class ParticleSystem;
	template <typename T> class NeighborQuery;

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

		SolidFluidInteraction(std::string name = "SolidFluidInteration");
		~SolidFluidInteraction() override;

	public:
		bool initialize() override;

		bool addRigidBody(std::shared_ptr<RigidBody<TDataType>> child);
		bool addParticleSystem(std::shared_ptr<ParticleSystem<TDataType>> child);

		bool resetStatus() override;

		void advance(Real dt) override;
	private:
		DeviceArrayField<Coord> m_position;

		DeviceArray<Real> m_mass;
		DeviceArray<int> m_objId;
		DeviceArray<Coord> m_vels;

		DeviceArray<Coord> posBuf;
		DeviceArray<Real> weights;
		DeviceArray<Coord> init_pos;

		std::shared_ptr<NeighborList<int>> m_nList;
		std::shared_ptr<NeighborQuery<TDataType>> m_nbrQuery;

		std::vector<std::shared_ptr<RigidBody<TDataType>>> m_rigids;
		std::vector<std::shared_ptr<ParticleSystem<TDataType>>> m_particleSystems;
	};


#ifdef PRECISION_FLOAT
	template class SolidFluidInteraction<DataType3f>;
#else
	template class SolidFluidInteraction<DataType3d>;
#endif
}