#pragma once
#include "Physika_Framework/Framework/FieldVar.h"
#include "Physika_Framework/Framework/FieldArray.h"
#include "Physika_Framework/Topology/FieldNeighbor.h"
#include "Physika_Dynamics/ParticleSystem/ElasticityModule.h"

namespace Physika {

	template<typename TDataType>
	class ElastoplasticityModule : public ElasticityModule<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef typename TPair<TDataType> NPair;

		ElastoplasticityModule();
		~ElastoplasticityModule() override {};
		
		bool constrain() override;

		void solvePlasticity();

		void reconstructRestShape();

		void RotateRestShape();

	protected:
		bool initializeImpl() override;

	private:
		VarField<Real> m_A;
		VarField<Real> m_B;

		DeviceArray<bool> m_bYield;
		DeviceArray<Matrix> m_invF;
		DeviceArray<Real> m_yiled_I1;
		DeviceArray<Real> m_yield_J2;
		DeviceArray<Real> m_I1;

	};


#ifdef PRECISION_FLOAT
	template class ElastoplasticityModule<DataType3f>;
#else
	template class ElastoplasticityModule<DataType3d>;
#endif
}