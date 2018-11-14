#pragma once
#include "Platform.h"
#include "Framework/Module.h"
#include "Physika_Core/Cuda_Array/Array.h"
#include "Physika_Core/DataTypes.h"
#include "Attribute.h"
#include "Kernel.h"
#include "Framework/FieldArray.h"

namespace Physika {
	template<typename TDataType>
	class ParticlePrediction : public Physika::Module
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticlePrediction();
		~ParticlePrediction() override {};
		
		bool execute() override;

		void PredictPosition(float dt);
		void PredictVelocity(float dt);

		void CorrectPosition(float dt);

		bool updateStates() override;

		virtual bool connectPosition(std::shared_ptr<Field>& pos) { return connect(pos, m_position); }
		virtual bool connectVelocity(std::shared_ptr<Field>& vel) { return connect(vel, m_velocity); }
		virtual bool connectAttribute(std::shared_ptr<Field>& att) { return connect(att, m_attribute); }
	private:
		Slot<DeviceBuffer<Coord>> m_position;
		Slot<DeviceBuffer<Coord>> m_velocity;
		Slot<DeviceBuffer<Attribute>> m_attribute;
	};

#ifdef PRECISION_FLOAT
	template class ParticlePrediction<DataType3f>;
#else
 	template class ParticlePrediction<DataType3d>;
#endif
}