#pragma once
#include "Dynamics/ParticleSystem/ElastoplasticityModule.h"

namespace PhysIKA {

	template<typename TDataType> class SummationDensity;

	template<typename TDataType>
	class GranularModule : public ElastoplasticityModule<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef TPair<TDataType> NPair;

		GranularModule();
		~GranularModule() override {};

	protected:
		bool initializeImpl() override;
		void computeMaterialStiffness() override;

	private:
		std::shared_ptr<SummationDensity<TDataType>> m_densitySum;
	};

#ifdef PRECISION_FLOAT
	template class GranularModule<DataType3f>;
#else
	template class GranularModule<DataType3d>;
#endif
}