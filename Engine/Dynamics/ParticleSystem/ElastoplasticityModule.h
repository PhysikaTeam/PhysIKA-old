#pragma once
#include "ElasticityModule.h"
#include "DensityPBD.h"

namespace Physika {

	template<typename TDataType>
	class ElastoplasticityModule : public ElasticityModule<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef TPair<TDataType> NPair;

		ElastoplasticityModule();
		~ElastoplasticityModule() override {};
		
		bool constrain() override;

		void solveElasticity() override;

		virtual void applyPlasticity();

		void applyYielding();

		void rotateRestShape();
		void reconstructRestShape();

		void setCohesion(Real c);
		void setFrictionAngle(Real phi);

		void enableFullyReconstruction();
		void disableFullyReconstruction();

		void enableIncompressibility();
		void disableIncompressibility();

	protected:
		bool initializeImpl() override;

		inline Real computeA()
		{
			Real phi = m_phi.getValue();
			return (Real)6.0*m_c.getValue()*cos(phi) / (3.0f + sin(phi)) / sqrt(3.0f);
		}


		inline Real computeB()
		{
			Real phi = m_phi.getValue();
			return (Real)2.0f*sin(phi) / (3.0f + sin(phi)) / sqrt(3.0f);
		}

	private:

		VarField<Real> m_c;
		VarField<Real> m_phi;

		VarField<bool> m_reconstuct_all_neighborhood;
		VarField<bool> m_incompressible;

		DeviceArray<bool> m_bYield;
		DeviceArray<Matrix> m_invF;
		DeviceArray<Real> m_yiled_I1;
		DeviceArray<Real> m_yield_J2;
		DeviceArray<Real> m_I1;

		std::shared_ptr<DensityPBD<TDataType>> m_pbdModule;
	};

#ifdef PRECISION_FLOAT
	template class ElastoplasticityModule<DataType3f>;
#else
	template class ElastoplasticityModule<DataType3d>;
#endif
}