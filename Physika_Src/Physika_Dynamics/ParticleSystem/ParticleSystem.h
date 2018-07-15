#ifndef FRAMEWORK_PARTICLESYSTEM_H
#define FRAMEWORK_PARTICLESYSTEM_H
#include "Physika_Core/DataTypes.h"
#include "Framework/Node.h"
#include "Framework/DeviceContext.h"
#include "Framework/Module.h"
#include "Attribute.h"
#include "INeighbors.h"
#include "Kernel.h"

namespace Physika {

	template<typename TDataType>
	class ParticleSystem : public Node
	{
	public:
 		typedef typename TDataType::Real Real;
 		typedef typename TDataType::Coord Coord;

		ParticleSystem(String name);
		virtual ~ParticleSystem() {};

		bool initialize() override;

		void advance(float dt) override;

		size_t GetParticleNumber() { return m_num->getValue(); }

		Real GetSamplingDistance() { return m_samplingDistance->getValue(); }

		void SetParticleNumber(size_t num);

		Real GetParticleMass() { return m_mass->getValue(); }

		Real GetSmoothingLength() { return m_smoothingLength->getValue(); }

		Real GetRestDensity() {
			return m_restDensity->getValue();
		}

		Real GetViscosity() {
			return m_viscosity->getValue();
		}
		void SetViscosity(Real vis) {
			m_viscosity->setValue(vis);
		}

		void SetGravity(Coord g) { m_gravity->setValue(g); }
		Coord GetGravity() { return m_gravity->getValue(); }

		void updateModules() override {};

		Coord GetLowerBound() { return m_lowerBound->getValue(); }
		Coord GetUpperBound() { return m_upperBound->getValue(); }

		DeviceBuffer<Coord>* AddNewPositionBuffer(DeviceBuffer<Coord>* buffer)
		{
			return NULL;
		}
		DeviceBuffer<Coord>* GetNewPositionBuffer()
		{
			SPtr< DeviceBuffer<Coord> > buf = getContext()->getDeviceBuffer<Coord>("POSITION");
			return buf.get();
		}

		DeviceBuffer<Coord>* AddOldPositionBuffer(DeviceBuffer<Coord>* buffer)
		{
			return NULL;
		};

		DeviceBuffer<Coord>* GetOldPositionBuffer() {
			SPtr< DeviceBuffer<Coord> > buf = getContext()->getDeviceBuffer<Coord>("OLD_POSITION");
			return buf.get();
		}

		DeviceBuffer<Coord>* AddNewVelocityBuffer(DeviceBuffer<Coord>* buffer) {
			return NULL;
		}
		DeviceBuffer<Coord>* GetNewVelocityBuffer() {
			SPtr< DeviceBuffer<Coord> > buf = getContext()->getDeviceBuffer<Coord>("VELOCITY");
			return buf.get();
		}

		DeviceBuffer<Coord>* AddOldVelocityBuffer(DeviceBuffer<Coord>* buffer)
		{
			return NULL;
		}
		DeviceBuffer<Coord>* GetOldVelocityBuffer()
		{
			SPtr< DeviceBuffer<Coord> > buf = getContext()->getDeviceBuffer<Coord>("OLD_VELOCITY");
			return buf.get();
		}

		DeviceBuffer<Real>* AddDensityBuffer(DeviceBuffer<Real>* buffer)
		{
			return NULL;
		}

		DeviceBuffer<Real>* GetDensityBuffer() {
			SPtr< DeviceBuffer<Real> > buf = getContext()->getDeviceBuffer<Real>("DENSITY");
			return buf.get();
		}

		DeviceBuffer<Attribute>* AddAttributeBuffer(DeviceBuffer<Attribute>* buffer) 
		{
			return NULL;
		}

		DeviceBuffer<Attribute>* GetAttributeBuffer();

		DeviceBuffer<NeighborList>* AddNeighborBuffer(DeviceBuffer<NeighborList>* buffer) {
			return NULL;
		}
		DeviceBuffer<NeighborList>* GetNeighborBuffer();

	private:
		HostVariablePtr<size_t> m_num;
		HostVariablePtr<Real> m_mass;
		HostVariablePtr<Real> m_smoothingLength;
		HostVariablePtr<Real> m_samplingDistance;
		HostVariablePtr<Real> m_restDensity;
		HostVariablePtr<Real> m_viscosity;

		HostVariablePtr<Coord> m_lowerBound;
		HostVariablePtr<Coord> m_upperBound;

		HostVariablePtr<Coord> m_gravity;
	};

#ifdef PRECISION_FLOAT
	template class ParticleSystem<DataType3f>;
#else
	template class ParticleSystem<DataType3d>;
#endif
}

#endif