#pragma once
#include "ParticleSystem.h"
#include "ParticleEmitter.h"

namespace PhysIKA
{
	/*!
	*	\class	ParticleFluid
	*	\brief	Position-based fluids.
	*
	*	This class implements a position-based fluid solver.
	*	Refer to Macklin and Muller's "Position Based Fluids" for details
	*
	*/
	template<typename TDataType>
	class ParticleEmitterRound : public ParticleEmitter<TDataType>
	{
		DECLARE_CLASS_1(ParticleEmitterRound, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleEmitterRound(std::string name = "particleEmitter");
		virtual ~ParticleEmitterRound();

		void setInfo(Coord pos, Coord dir, Real r ,Real distance);
		
		void gen_random() override;
		
		
		//void advance(Real dt) override;
	private:
		

		//DEF_NODE_PORTS(ParticleSystems, ParticleSystem<TDataType>, "Particle Systems");
	};

#ifdef PRECISION_FLOAT
	template class ParticleEmitterRound<DataType3f>;
#else
	template class ParticleEmitterRound<DataType3d>;
#endif
}