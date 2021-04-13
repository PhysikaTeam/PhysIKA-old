#pragma once
#include "ParticleSystem.h"

namespace PhysIKA
{
	/*!
	*	\class	ParticleEimitter
	*	\brief	
	*
	*	
	*	
	*
	*/
	template <typename T> class ParticleFluid;
	template<typename TDataType>
	class ParticleEmitter : public ParticleSystem<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleEmitter(std::string name = "particle emitter");
		virtual ~ParticleEmitter();

		void advance2(Real dt);
		void advance(Real dt) override;
		virtual void generateParticles();

		void updateTopology() override;
		bool resetStatus() override;


		//DEF_VAR(Centre, Vector3f, 0, "Emitter location");
		//DEF_VAR(Radius, Real, 0.1, "Emitter scale");
		DEF_VAR(VelocityMagnitude, Real, 1, "Emitter Velocity");
		DEF_VAR(SamplingDistance, Real, 0.005, "Emitter Sampling Distance");

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