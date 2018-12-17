#pragma once
#include "Physika_Framework/Framework/NumericalIntegrator.h"

namespace Physika {
	template<typename TDataType>
	class ParticleIntegrator : public NumericalIntegrator
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleIntegrator();
		~ParticleIntegrator() override {};
		
		void begin() override;
		void end() override;

		bool integrate() override;

		bool updateVelocity();
		bool updatePosition();

	private:

	};

#ifdef PRECISION_FLOAT
	template class ParticleIntegrator<DataType3f>;
#else
 	template class ParticleIntegrator<DataType3d>;
#endif
}