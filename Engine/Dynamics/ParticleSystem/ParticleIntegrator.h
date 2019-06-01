#pragma once
#include "Framework/Framework/NumericalIntegrator.h"
#include "Framework/Framework/FieldVar.h"
#include "Framework/Framework/FieldArray.h"

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

	protected:
		bool initializeImpl() override;

	public:
		DeviceArrayField<Coord> m_position;
		DeviceArrayField<Coord> m_velocity;
		DeviceArrayField<Coord> m_forceDensity;

	private:
		DeviceArray<Coord> m_prePosition;
		DeviceArray<Coord> m_preVelocity;
	};

#ifdef PRECISION_FLOAT
	template class ParticleIntegrator<DataType3f>;
#else
 	template class ParticleIntegrator<DataType3d>;
#endif
}