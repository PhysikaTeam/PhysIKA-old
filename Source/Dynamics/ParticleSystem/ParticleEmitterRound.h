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
		
		void gen_random();
		
		void advance(Real dt) override;
	private:
		Real radius;
		Real sampling_distance;
		Coord centre;
		Coord dir;

		DeviceArray<Coord> gen_pos;
		DeviceArray<Coord> gen_vel;
		
		DeviceArray<Coord> pos_buf;
		DeviceArray<Coord> vel_buf;
		DeviceArray<Coord> force_buf;
		int sum = 0;

	};

#ifdef PRECISION_FLOAT
	template class ParticleEmitterRound<DataType3f>;
#else
	template class ParticleEmitterRound<DataType3d>;
#endif
}