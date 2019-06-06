#pragma once
#include "Framework/Framework/FieldVar.h"
#include "Framework/Framework/FieldArray.h"
#include "Framework/Topology/FieldNeighbor.h"
#include "Framework/Framework/ModuleConstraint.h"
#include "DensityPBD.h"

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
		~ElasticityModule() override;
		
		bool constrain() override;

		void solveElasticity();

		void takeOneIteration();

		void resetRestShape();

		void setMu(Real mu) { m_mu.setValue(mu); };
		void setLambda(Real lambda) { m_lambda.setValue(lambda); }

		void setHorizon(Real len) { m_horizon.setValue(len); }
		void setIterationNumber(int num) { m_interation = num; }

		DeviceArray<Real>& getDensity()
		{
			return m_pbdModule->m_density.getValue();
		}

	protected:
		bool initializeImpl() override;

	public:
		/**
		 * @brief Horizon
		 * A positive number represents the radius of neighborhood for each point
		 */
		VarField<Real> m_horizon;
		/**
		 * @brief Sampling distance
		 * Indicate the sampling distance when particles are created.
		 */
		VarField<Real> m_distance;

		/**
		 * @brief Particle position
		 */
		DeviceArrayField<Coord> m_position;
		/**
		 * @brief Particle velocity
		 */
		DeviceArrayField<Coord> m_velocity;

		/**
		 * @brief Neighboring particles
		 * 
		 */
		NeighborField<int> m_neighborhood;
		NeighborField<NPair> m_restShape;

	protected:
		/**
		* @brief Lame parameters
		* m_lambda controls the isotropic part while mu controls the deviatoric part.
		*/
		VarField<Real> m_mu;
		VarField<Real> m_lambda;

		DeviceArray<Real> m_bulkCoefs;

	private:
		int m_interation = 3;

		DeviceArray<Real> m_stiffness;
		DeviceArray<Real> m_lambdas;

		DeviceArray<Coord> m_accPos;
		DeviceArray<Matrix> m_invK;
		
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