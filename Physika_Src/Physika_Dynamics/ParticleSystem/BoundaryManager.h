#pragma once

#include <vector>

#include "Physika_Core/Cuda_Array/Array.h"
#include "Physika_Geometry/SDF/DistanceField3D.h"
#include "ParticleSystem.h"
#include "Framework/Module.h"

namespace Physika {

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
	class BoundaryManager : public Module
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		BoundaryManager(ParticleSystem<TDataType>* parent = NULL);
		~BoundaryManager(void);

		bool execute() override;

		void Constrain(DeviceArray<Coord>& posArr, DeviceArray<Coord>& velArr, DeviceArray<Attribute>& attArr, float dt);

		void InsertBarrier(Barrier<TDataType> *in_barrier) {
			m_barriers.push_back(in_barrier);
		}

		inline int size() const {
			return (int)m_barriers.size();
		}

// 		static BoundaryManager* Create(ParticleSystem<TDataType>* parent, DeviceType deviceType = DeviceType::GPU)
// 		{
// 			return new BoundaryManager(parent, deviceType);
// 		}

	public:
		ParticleSystem<TDataType>* m_parent;
		std::vector<Barrier<TDataType> *> m_barriers;

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