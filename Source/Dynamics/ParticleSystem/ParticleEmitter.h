#pragma once
#include "ParticleSystem.h"

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
	template <typename T> class ParticleFluid;
	template<typename TDataType>
	class ParticleEmitter : public ParticleSystem<TDataType>
	{
		DECLARE_CLASS_1(ParticleEmitter, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleEmitter(std::string name = "particle emitter");
		virtual ~ParticleEmitter();
		bool addOutput(std::shared_ptr<ParticleFluid<TDataType>> child, std::shared_ptr<ParticleSystem<TDataType>> self);
		void getRotMat(Coord rot);
		void advance(Real dt) override;
		virtual void gen_random();

		Real radius;
		Real sampling_distance;
		Coord centre;
		Coord dir;

		Coord axis;
		Real angle;

		DeviceArray<Coord> gen_pos;
		DeviceArray<Coord> gen_vel;

		DeviceArray<Coord> pos_buf;
		DeviceArray<Coord> vel_buf;
		DeviceArray<Coord> force_buf;
		int sum = 0;
	private:
		
	};

#ifdef PRECISION_FLOAT
	template class ParticleEmitter<DataType3f>;
#else
	template class ParticleEmitter<DataType3d>;
#endif
}