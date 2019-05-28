#pragma once
#include "Physika_Framework/Framework/FieldVar.h"
#include "Physika_Framework/Framework/FieldArray.h"
#include "Physika_Framework/Topology/FieldNeighbor.h"
#include "Physika_Framework/Framework/ModuleConstraint.h"
#include "DensityPBD.h"

namespace Physika {

	template<typename TDataType>
	class ElasticityModule : public ConstraintModule
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef typename TPair<TDataType> NPair;

		ElasticityModule();
		~ElasticityModule() override {};
		
		bool constrain() override;

		void solveElasticity();

		void constructRestShape(NeighborList<int>& nbr, DeviceArray<Coord>& pos);


		void setHorizon(Real len) { m_horizon = len; }

		DeviceArray<Real>& getDensity()
		{
			return m_pbdModule->m_density.getValue();
		}

	protected:
		bool initializeImpl() override;


	public:
		VarField<Real> m_horizon;
		VarField<Real> m_distance;

		VarField<Real> m_mu;
		VarField<Real> m_lambda;

		DeviceArrayField<Coord> m_position;
		DeviceArrayField<Coord> m_velocity;

		NeighborField<int> m_neighborhood;
		NeighborField<NPair> m_restShape;

	protected:
		DeviceArray<Real> m_bulkCoefs;

	private:
		DeviceArray<Real> m_stiffness;
		DeviceArrayField<Real>* m_lambdas;
		
		DeviceArrayField<Coord>* m_tmpPos;
		DeviceArrayField<Coord>* m_accPos;
		DeviceArrayField<Matrix>* m_invK;
		DeviceArray<Matrix> m_F;

		DeviceArray<Coord> m_oldPosition;

		std::shared_ptr<DensityPBD<TDataType>> m_pbdModule;
	};

#ifdef PRECISION_FLOAT
	template class ElasticityModule<DataType3f>;
#else
	template class ElasticityModule<DataType3d>;
#endif
}