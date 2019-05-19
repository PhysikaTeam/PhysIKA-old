#pragma once
#include "Physika_Framework/Framework/FieldVar.h"
#include "Physika_Framework/Framework/FieldArray.h"
#include "Physika_Framework/Topology/FieldNeighbor.h"
#include "Physika_Framework/Framework/ModuleConstraint.h"

namespace Physika {

	template<typename TDataType>
	class ElasticityModule : public ConstraintModule
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef TPair<TDataType> NPair;

		ElasticityModule();
		~ElasticityModule() override {};
		
		bool constrain() override;

		void constructRestConfiguration(NeighborList<int>& nbr, DeviceArray<Coord>& pos);
		bool isUpdateRequired() { return m_needUpdate; }

		void setHorizon(Real len) { m_horizon = len; }

	protected:
		bool initializeImpl() override;


	public:
		VarField<Real> m_horizon;

		DeviceArrayField<Coord> m_position;
		DeviceArrayField<Coord> m_velocity;

		NeighborField<int> m_neighborhood;
		NeighborField<NPair> m_referenceConfiguration;

	private:
		bool m_needUpdate;

		NeighborList<NPair> m_refPos;

		DeviceArrayField<Real>* m_lambdas;
		DeviceArrayField<Real>* m_bulkCoef;
		DeviceArrayField<Coord>* m_tmpPos;
		DeviceArrayField<Coord>* m_accPos;
		DeviceArrayField<Matrix>* m_refMatrix;

		DeviceArray<Coord> m_oldPosition;
	};


#ifdef PRECISION_FLOAT
	template class ElasticityModule<DataType3f>;
#else
	template class ElasticityModule<DataType3d>;
#endif
}