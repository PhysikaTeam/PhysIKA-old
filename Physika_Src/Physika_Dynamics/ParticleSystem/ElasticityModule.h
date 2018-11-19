#pragma once
#include "Physika_Core/Platform.h"
#include "Kernel.h"
#include "glm/mat3x3.hpp"
#include "Physika_Core/Cuda_Array/Array.h"
#include "Physika_Core/DataTypes.h"
#include "Physika_Framework/Framework/ModuleConstraint.h"
#include "Kernel.h"
#include "Physika_Framework/Topology/INeighbors.h"
#include "Physika_Framework/Framework/FieldVar.h"
#include "Physika_Framework/Framework/FieldArray.h"

namespace Physika {

	template<typename TDataType>
	class ElasticityModule : public ConstraintModule
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef typename TRestShape<TDataType> RestShape;

		ElasticityModule();
		~ElasticityModule() override {};
		
		bool execute() override;

		bool connectPosition(std::shared_ptr<Field>& pos) override { return connect(pos, m_position); }
		bool connectVelocity(std::shared_ptr<Field>& vel) override { return connect(vel, m_velocity); }

		bool connectRadius(std::shared_ptr<Field>& radius) { return connect(radius, m_radius); }
		bool connectSamplingDistance(std::shared_ptr<Field>& distance) { return connect(distance, m_samplingDistance); }
		bool connectState(std::shared_ptr<Field>& state) { return connect(state, m_state); }
		bool connectPrePosition(std::shared_ptr<Field>& prePos) { return connect(prePos, m_prePosition); }
		bool connectInitPosition(std::shared_ptr<Field>& initPos) { return connect(initPos, m_initPosition); }
		bool connectRestShape(std::shared_ptr<Field>& restShape) { return connect(restShape, m_restShape); }

	private:
		DeviceArrayField<Real>* m_lambdas;
		DeviceArrayField<Real>* m_bulkCoef;
		DeviceArrayField<Coord>* m_tmpPos;
		DeviceArrayField<Coord>* m_accPos;
		DeviceArrayField<Matrix>* m_refMatrix;

		Slot<HostVarField<Real>>  m_radius;
		Slot<HostVarField<Real>>  m_samplingDistance;
		Slot<DeviceArrayField<int>>  m_state;
		Slot<DeviceArrayField<Coord>> m_position;
		Slot<DeviceArrayField<Coord>> m_prePosition;
		Slot<DeviceArrayField<Coord>> m_initPosition;
		Slot<DeviceArrayField<Coord>>  m_velocity;
		Slot<DeviceArrayField<RestShape>> m_restShape;
	};


#ifdef PRECISION_FLOAT
	template class ElasticityModule<DataType3f>;
#else
	template class DensityPBD<DataType3d>;
#endif
}