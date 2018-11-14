#pragma once

#include <vector>

#include "Physika_Core/Cuda_Array/Array.h"
#include "Physika_Geometry/SDF/DistanceField3D.h"
#include "Attribute.h"
#include "cuda_runtime.h"
#include "Framework/ModuleConstraint.h"

namespace Physika {

//	template <typename> class ParticleSystem;

	template<typename TDataType>
	class Barrier
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		Barrier() { normalFriction = 0.95f; tangentialFriction = 0.1f; }
		~Barrier() {};

		//		virtual void Inside(const float3& in_pos) const;
		virtual void Constrain(DeviceArray<Coord>& posArr, DeviceArray<Coord>& velArr, DeviceArray<int>& attArr, Real dt) {};

		void SetNormalFriction(float val) { normalFriction = val; }
		void SetTangentialFriction(float val) { tangentialFriction = val; }

	protected:
		Real normalFriction;
		Real tangentialFriction;
	};

	template<typename TDataType>
	class BarrierDistanceField3D : public Barrier<TDataType> {

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		// CONVENTION: normal n should point outwards, i.e., away from inside
		// of constraint
		BarrierDistanceField3D(DistanceField3D<TDataType> *df) :
			Barrier(), distancefield3d(df) {
		}

		~BarrierDistanceField3D() { distancefield3d->Release(); }
		// 
		// 		virtual void GetDistanceAndNormal(const float3 &p, float &dist, float3 &normal) const {
		// 			distancefield3d->GetDistance(p, dist, normal);
		// 		}

		void Constrain(DeviceArray<Coord>& posArr, DeviceArray<Coord>& velArr, DeviceArray<int>& attArr, Real dt) override;

		DistanceField3D<TDataType> * distancefield3d;
	};


	template<typename TDataType>
	class BoundaryManager : public ConstraintModule
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		BoundaryManager();
		~BoundaryManager(void);

		bool execute() override;

		void Constrain(DeviceArray<Coord>& posArr, DeviceArray<Coord>& velArr, DeviceArray<Attribute>& attArr, float dt);

		void InsertBarrier(Barrier<TDataType> *in_barrier) {
			m_barriers.push_back(in_barrier);
		}

		inline int size() const {
			return (int)m_barriers.size();
		}

		virtual bool connectPosition(std::shared_ptr<Field>& pos) { return connect(pos, m_position); }
		virtual bool connectVelocity(std::shared_ptr<Field>& vel) { return connect(vel, m_velocity); }
		virtual bool connectAttribute(std::shared_ptr<Field>& att) { return connect(att, m_attribute); }

// 		static BoundaryManager* Create(ParticleSystem<TDataType>* parent, DeviceType deviceType = DeviceType::GPU)
// 		{
// 			return new BoundaryManager(parent, deviceType);
// 		}

	public:
		std::vector<Barrier<TDataType> *> m_barriers;

		Slot<DeviceBuffer<Coord>> m_position;
		Slot<DeviceBuffer<Coord>> m_velocity;
		Slot<DeviceBuffer<Attribute>> m_attribute;

		DeviceBuffer<int>* m_bConstrained;
	};

#ifdef PRECISION_FLOAT
	template class BoundaryManager<DataType3f>;
	template class BarrierDistanceField3D<DataType3f>;
#else
	template class BoundaryManager<DataType3d>;
	template class BarrierDistanceField3D<DataType3d>;
#endif
}