#pragma once
#include "Dynamics/ParticleSystem/ElastoplasticityModule.h"

namespace Physika {

	template<typename TDataType> class DensitySummation;

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
		std::shared_ptr<DensitySummation<TDataType>> m_densitySum;
	};

#ifdef PRECISION_FLOAT
	template class GranularModule<DataType3f>;
#else
	template class GranularModule<DataType3d>;
#endif
}